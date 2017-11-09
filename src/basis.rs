//! Basis manipulation.
//!
//! We make heavy use of block-diagonal matrices here.  Conceptually, a
//! block-diagonal matrix is equivalent to a list of blocks, which need not be
//! the same size nor do they have to be square.  Many operations on
//! block-diagonal matrices translate to just operations on each block, which
//! can be easily parallelized in principle.
//!
//! - "Channels" are labels that uniquely identifies each block.
//! - "Auxiliary" are labels that identify each row or column within a block.
//!
//! The combination of channel and auxiliary must, by definition, uniquely
//! identify a state.
//!
//! We further subdivide each block into subblocks.  This is needed for, say,
//! separating the particle vs hole states.
//!
//! - "Parts" are labels that identify the subblocks along one of the axes.
//!
//! The use of parts has both pros and cons.  The advantage is that things
//! like diagonalization of a block matrix is very simple because each block
//! can be diagonalized independently.  The disadvantage is that it adds
//! another layer of complication and overhead.
//!
//! Naming convention:
//!
//!   - `l` = channel (in text, `λ` is sometimes used)
//!   - `u` = auxiliary
//!   - `x` = part ("eXcitation" / unoccupancy)
//!   - `p` = state, isomorphic to `(l, u)`
//!
//! These letters could refer to either indices or the concrete objects.  When
//! there is ambiguity, indicial variables are prefixed with `i`.
//!
//! Be aware that these labels may also have other meanings in more concrete
//! contexts (e.g. `l` can mean orbital angular momentum, and `p` can mean
//! parity).
//!
use std::{fmt, iter, mem, vec};
use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use fnv::FnvHashMap;
use super::block_matrix::{BlockMat, BlockMatMut};
use super::matrix::{Mat, MatShape, Matrix};
use super::cache2::Cache;
use super::utils;

pub type State2<T> = (T, T);

lazy_static! {
    /// Global cache for storing basis information.
    pub static ref CACHE: Cache = Cache::default();
}

pub fn siphash128<T: Hash + ?Sized>(value: &T, key: (u64, u64)) -> (u64, u64) {
    use siphasher::sip128::{Hash128, Hasher128, SipHasher};
    let mut hasher = SipHasher::new_with_keys(key.0, key.1);
    value.hash(&mut hasher);
    let Hash128 { h1, h2 } = hasher.finish128();
    (h1, h2)
}

#[derive(Clone, Debug)]
pub struct Hashed<T> {
    pub inner: T,
    pub hash: (u64, u64),
}

impl<T: Hash> Hashed<T> {
    pub fn new(inner: T) -> Self {
        let hash = siphash128(&inner, (0, 0));
        Hashed { inner, hash }
    }
}

impl<T> PartialEq for Hashed<T> {
    fn eq(&self, other: &Self) -> bool {
        self.hash.eq(&other.hash)
    }
}

impl<T> Eq for Hashed<T> {}

impl<T> Hash for Hashed<T> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.hash.hash(hasher)
    }
}

impl<T> Deref for Hashed<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Hashed<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub trait IntoUsize {
    fn into_usize(self) -> usize;
}

impl IntoUsize for u32 {
    fn into_usize(self) -> usize { self as _ }
}

impl IntoUsize for usize {
    fn into_usize(self) -> usize { self }
}

pub trait FromUsize {
    fn from_usize(i: usize) -> Self;
}

impl FromUsize for u32 {
    fn from_usize(i: usize) -> Self { utils::cast(i) }
}

impl FromUsize for usize {
    fn from_usize(i: usize) -> Self { i }
}

#[derive(Clone, Copy, Debug)]
pub struct HashChartInsertResult<I = usize> {
    /// Whether the insert was successful
    pub inserted: bool,
    /// Index of the inserted item
    pub index: I,
}

/// A chart is a bijection between a zero-based interval (i.e. indices) and an
/// arbitrary set of objects of type `T`.  `HashChart` is an implementation of
/// this idea using `HashMap`.
#[derive(Clone)]
pub struct HashChart<T, I = usize> {
    pub encoder: FnvHashMap<T, I>,
    pub decoder: Vec<T>,
}

impl<T, I> Default for HashChart<T, I> {
    fn default() -> Self {
        Self {
            encoder: utils::default_hash_map(),
            decoder: Default::default(),
        }
    }
}

impl<T: fmt::Debug, I> fmt::Debug for HashChart<T, I> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("HashChart").field(&self.decoder).finish()
    }
}

impl<T, I> iter::FromIterator<T> for HashChart<T, I> where
    T: Hash + Eq + Clone,
    I: FromUsize + Clone,
{
    fn from_iter<U: IntoIterator<Item = T>>(iter: U) -> Self {
        let mut chart = HashChart::default();
        for t in iter {
            chart.insert(t);
        }
        chart
    }
}

impl<T, I> IntoIterator for HashChart<T, I> {
    type Item = T;
    type IntoIter = vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.decoder.into_iter()
    }
}

impl<T: Hash + Eq, I> HashChart<T, I> {
    pub fn encode<Q>(&self, q: &Q) -> Option<&I>
        where Q: Hash + Eq,
              T: Borrow<Q>,
    {
        self.encoder.get(q)
    }
}

impl<T, I: FromUsize> HashChart<T, I> {
    pub fn len(&self) -> I {
        I::from_usize(self.decoder.len())
    }
}

impl<T, I: IntoUsize> HashChart<T, I> {
    pub fn decode(&self, i: I) -> Option<&T> {
        self.decoder.get(i.into_usize())
    }
}

impl<T: Hash + Eq, I: FromUsize> HashChart<T, I> {
    /// Reorder the elements in the HashChart via some total ordering,
    /// changing the association between elements and indices.  The callback
    /// receives the permutation that maps from old indices to new.
    pub fn reorder_by_key<K, F, G>(&mut self, key: F, mut permut: G) where
        K: Ord,
        F: FnMut(&T) -> K,
        G: FnMut(&I, &I),
    {
        self.decoder.sort_by_key(key);
        for (i, t) in self.decoder.iter().enumerate() {
            let i = I::from_usize(i);
            let j = self.encoder.get_mut(t).expect("HashChart is corrupt");
            permut(j, &i);
            *j = i;
        }
    }
}

impl<T: Hash + Eq + Clone, I: FromUsize + Clone> HashChart<T, I> {
    pub fn insert(&mut self, t: T) -> HashChartInsertResult<I> {
        let mut inserted = false;
        let query = t.clone();
        let decoder = &mut self.decoder;
        let index = self.encoder.entry(query).or_insert_with(|| {
            inserted = true;
            let index = decoder.len();
            decoder.push(t);
            I::from_usize(index)
        }).clone();
        HashChartInsertResult { inserted, index }
    }
}

/// A basis layout has only enough information to understand the structure of
/// a matrix along one axis, but not to interpret its elements.
#[derive(Clone, Debug, Default)]
pub struct BasisLayout {
    pub num_parts: u32,
    /// `part_offsets[l * num_parts + x - 1] == ∑[x' < x] num_states_l_x(l, x')
    /// where l < nl && x > 0 && x < num_parts`
    pub part_offsets: Box<[u32]>,
    /// `chan_offsets[l] == ∑[l' < l] num_states_l(l')
    /// where l ≤ nl`
    pub chan_offsets: Box<[u32]>,
    /// `states[p] == l
    /// where p ≤ np`
    pub state_chans: Box<[u32]>,
}

impl BasisLayout {
    pub fn num_states(&self) -> u32 {
        self.state_chans.len() as _
    }

    pub fn num_chans(&self) -> u32 {
        self.chan_offsets.len() as u32 - 1
    }

    pub fn num_parts(&self) -> u32 {
        self.num_parts
    }

    pub fn num_auxs(&self, l: u32) -> u32 {
        self.chan_offset(l + 1) - self.chan_offset(l)
    }

    pub fn chan_dim(&self, l: u32) -> u32 {
        let n = self.chan_offset(l + 1) - self.chan_offset(l);
        debug_assert_eq!(self.part_offset(l, self.num_parts), n);
        n
    }

    pub fn part_dim(&self, l: u32, x: u32) -> u32 {
        self.part_offset(l, x + 1) - self.part_offset(l, x)
    }

    pub fn channelize(&self, p: u32) -> ChanState {
        let l = self.state_chans[p as usize];
        let u = p - self.chan_offset(l);
        ChanState { l, u }
    }

    pub fn dechannelize(&self, lu: ChanState) -> u32 {
        self.chan_offset(lu.l) + lu.u
    }

    pub fn chan_offset(&self, l: u32) -> u32 {
        self.chan_offsets[l as usize]
    }

    pub fn chan_offsets(&self) -> &[u32] {
        &self.chan_offsets
    }

    pub fn part_offset(&self, l: u32, x: u32) -> u32 {
        if x == 0 {
            0
        } else if x == self.num_parts() {
            self.num_auxs(l)
        } else {
            *self.part_offsets().get(l as _, (x - 1) as _).unwrap()
        }
    }

    pub fn part_offsets(&self) -> Mat<u32> {
        Mat::new(
            &mut (&self.part_offsets as _),
            MatShape::packed(
                self.num_chans() as _,
                (self.num_parts - 1) as _,
            ).validate().unwrap(),
        ).unwrap()
    }
}

/// A basis chart contains information needed to interpret an axis of a
/// block-diagonal matrix.  The type is parametrized by:
///
///   - `L` (`λ`): channel type,
///   - `X` (`χ`): part type, and
///   - `U` (`μ`): auxiliary type.
///
#[derive(Clone)]
pub struct BasisChart<L, X, U> {
    pub chan_chart: HashChart<L, u32>,
    pub part_chart: HashChart<X, u32>,
    /// `aux_encoder[(l, μ)] == u`
    pub aux_encoder: FnvHashMap<(u32, U), u32>,
    /// `aux_decoder[chan_offset(l) + u] == (L, U)`
    pub aux_decoder: Box<[U]>,
    pub layout: BasisLayout,
}

impl<L, X, U> Default for BasisChart<L, X, U> {
    fn default() -> Self {
        Self {
            chan_chart: Default::default(),
            part_chart: Default::default(),
            aux_encoder: utils::default_hash_map(),
            aux_decoder: Default::default(),
            layout: Default::default(),
        }
    }
}

impl<L, X, U> fmt::Debug for BasisChart<L, X, U>
    where L: fmt::Debug,
          X: fmt::Debug,
          U: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("BasisChart")
            .field("chan_chart", &self.chan_chart)
            .field("part_chart", &self.part_chart)
            .field("aux_encoder", &utils::DebugWith(|f: &mut fmt::Formatter| {
                f.write_str("n/a")
            }))
            .field("aux_decoder", &self.aux_decoder)
            .field("layout", &self.layout)
            .finish()
    }
}

impl<L, X, U> fmt::Display for BasisChart<L, X, U>
    where L: fmt::Display,
          X: fmt::Display,
          U: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for il in 0 .. self.layout.num_chans() {
            let l = self.decode_chan(il).unwrap();
            let ol = self.layout.chan_offset(il);
            writeln!(f, "[{}] {} (offset={})", il, l, ol)?;
            for ix in 0 .. self.layout.num_parts {
                let x = self.decode_part(ix).unwrap();
                writeln!(f, "  [{}] {}", ix, x)?;
                let u_start = self.layout.part_offset(il, ix);
                let u_end = self.layout.part_offset(il, ix + 1);
                for iu in u_start .. u_end {
                    let u = self.decode_aux(il, iu).unwrap();
                    writeln!(f, "    [{}] {}", iu, u)?;
                }
            }
        }
        let nl = self.layout.num_chans();
        writeln!(f, "[{}] <END> [offset={}]", nl, self.layout.chan_offset(nl))
    }
}

impl<L, X, U> BasisChart<L, X, U>
    where L: Hash + Eq + Clone,
          X: Hash + Eq + Clone,
          U: Hash + Eq + Clone,
{
    pub fn new<I>(states: I) -> Self where
        I: Iterator<Item = PartState<X, ChanState<L, U>>>,
    {
        Self::new_with(states, Default::default(), Default::default())
    }

    pub fn new_with<I>(
        states: I,
        mut chan_chart: HashChart<L, u32>,
        mut part_chart: HashChart<X, u32>,
    ) -> Self where
        I: Iterator<Item = PartState<X, ChanState<L, U>>>,
    {
        let mut state_chart = HashChart::default(); // this one is temporary
        let mut lxps = Vec::default();
        for PartState { x, p: ChanState { l, u } } in states {
            let l = chan_chart.insert(l).index;
            let x = part_chart.insert(x).index;
            let inserted_lu = state_chart.insert((l, u));
            assert!(inserted_lu.inserted, "every state (l, u) must be unique");
            let p: u32 = inserted_lu.index;
            lxps.push((l, x, p));
        }
        lxps.sort_unstable();
        let num_chans = chan_chart.len();
        let num_parts = part_chart.len();
        let mut part_offsets = Vec::default();
        let mut chan_offsets = vec![0];
        let mut state_chans = Vec::default();
        let mut aux_encoder = FnvHashMap::default();
        let mut aux_decoder = Vec::default();
        let mut lxps = lxps.iter();
        for l in 0 .. num_chans {
            let mut u = 0;
            for x in 0 .. num_parts {
                loop {
                    let mut lxps_i = lxps.clone();
                    match lxps_i.next() {
                        Some(&(l_i, x_i, p_i)) if l == l_i && x == x_i => {
                            lxps = lxps_i;
                            let state = state_chart.decode(p_i).unwrap().clone();
                            aux_decoder.push(state.1.clone());
                            aux_encoder.insert(state, u);
                            state_chans.push(l_i);
                            u += 1;
                        }
                        _ => break,
                    }
                }
                if x != num_parts - 1 {
                    part_offsets.push(u);
                }
            }
            chan_offsets.push(utils::cast(aux_decoder.len()));
        }
        Self {
            chan_chart,
            part_chart,
            aux_encoder,
            aux_decoder: aux_decoder.into_boxed_slice(),
            layout: BasisLayout {
                num_parts,
                part_offsets: part_offsets.into_boxed_slice(),
                chan_offsets: chan_offsets.into_boxed_slice(),
                state_chans: state_chans.into_boxed_slice(),
            },
        }
    }
}

impl<L, X, U> BasisChart<L, X, U> {
    pub fn layout(&self) -> &BasisLayout {
        &self.layout
    }

    pub fn decode_chan(&self, l: u32) -> Option<&L> {
        self.chan_chart.decode(l)
    }

    pub fn decode_part(&self, x: u32) -> Option<&X> {
        self.part_chart.decode(x)
    }

    pub fn decode_aux(&self, l: u32, u: u32) -> Option<&U> {
        self.aux_decoder.get((self.layout.chan_offset(l) + u) as usize)
    }
}

impl<L: Hash + Eq, X, U> BasisChart<L, X, U> {
    pub fn encode_chan(&self, l: &L) -> Option<u32> {
        self.chan_chart.encode(l).cloned()
    }
}

impl<L, X: Hash + Eq, U> BasisChart<L, X, U> {
    pub fn encode_part(&self, x: &X) -> Option<u32> {
        self.part_chart.encode(x).cloned()
    }
}

impl<L, X, U: Hash + Eq> BasisChart<L, X, U> {
    pub fn encode_aux(&self, l: u32, u: &U) -> Option<u32> {
        utils::with_tuple2_ref(&l, u, |lu| {
            self.aux_encoder.get(lu).cloned()
        })
    }
}

#[derive(Clone, Debug)]
pub struct MatLayout<'a> {
    pub block_offsets: Vec<usize>,
    pub block_strides: &'a BasisLayout,
    pub left: &'a BasisLayout,
    pub right: &'a BasisLayout,
}

impl<'a> MatLayout<'a> {
    pub fn new(left: &'a BasisLayout, right: &'a BasisLayout) -> Self {
        let mut block_offsets = vec![0];
        let mut i = 0;
        assert_eq!(left.num_chans(), right.num_chans());
        for l in 0 .. left.num_chans() {
            i += (left.chan_dim(l) as usize)
                * (right.chan_dim(l) as usize);
            block_offsets.push(i);
        }
        MatLayout {
            block_offsets,
            block_strides: right,
            left,
            right,
        }
    }

    pub fn len(&self) -> usize {
        *self.block_offsets.last().unwrap()
    }

    pub fn block_dim(&self, l: u32) -> usize {
        self.block_offsets[(l + 1) as usize] - self.block_offsets[l as usize]
    }

    pub fn offset(&self, l: u32, u1: u32, u2: u32) -> usize {
        debug_assert!((l as usize) < self.block_offsets.len() - 1);
        debug_assert!(u1 < self.left.chan_dim(l));
        debug_assert!(u2 < self.right.chan_dim(l));
        self.block_offsets[l as usize]
            + u1 as usize * self.block_strides.chan_dim(l) as usize
            + u2 as usize
    }
}

#[derive(Debug)]
pub struct MatChart<'a, L: 'a, X1: 'a, X2: 'a, U1: 'a, U2: 'a> {
    pub layout: &'a MatLayout<'a>,
    pub left: &'a BasisChart<L, X1, U1>,
    pub right: &'a BasisChart<L, X2, U2>,
}

impl<'a, L, X1, X2, U1, U2> Clone for MatChart<'a, L, X1, X2, U1, U2> {
    fn clone(&self) -> Self { *self }
}

impl<'a, L, X1, X2, U1, U2> Copy for MatChart<'a, L, X1, X2, U1, U2> {}

impl<'a, L, X1, X2, U1, U2> MatChart<'a, L, X1, X2, U1, U2>
    where L: Hash + Eq,
          U1: Hash + Eq,
          U2: Hash + Eq,
{
    pub fn offset(self, l: &L, u1: &U1, u2: &U2) -> usize {
        let il = self.left.encode_chan(&l).expect("invalid channel") as u32;
        let iu1 = self.left.encode_aux(il, &u1).expect("invalid left state") as u32;
        let iu2 = self.right.encode_aux(il, &u2).expect("invalid right state") as u32;
        self.layout.offset(il, iu1, iu2)
    }
}

/// Orbital index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Orb(pub u32);

/// Channelized state index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChanState<L = u32, U = u32> {
    /// Channel index
    pub l: L,
    /// Auxiliary index
    pub u: U,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PartState<X, P> {
    pub x: X,
    pub p: P,
}

/// Associates data types with sequences.
pub trait Increment: Sized {
    /// Given the current item, return the next item in the associated
    /// sequence.
    fn increment(&self) -> Option<Self>;
}

/// Iterator over `Option<impl Increment>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fence<T>(pub T);

impl<T: Increment> Iterator for Fence<Option<T>> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let t_new = self.0.as_ref().and_then(Increment::increment);
        mem::replace(&mut self.0, t_new)
    }
}

/// Occupancy of an orbital (single-particle state).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Occ {
    /// Occupied ("hole") orbital
    I,
    /// Unoccupied ("particle") orbital
    A,
}

impl From<Occ> for usize {
    fn from(x: Occ) -> Self {
        use self::Occ::*;
        match x {
            I => 0,
            A => 1,
        }
    }
}

impl From<Occ> for u32 {
    fn from(x: Occ) -> Self {
        usize::from(x) as _
    }
}

impl From<bool> for Occ {
    fn from(x: bool) -> Self {
        match x {
            false => Occ::I,
            true => Occ::A,
        }
    }
}

impl Increment for Occ {
    fn increment(&self) -> Option<Self> {
        Self::from_usize(usize::from(*self) + 1)
    }
}

impl Occ {
    pub fn chart() -> HashChart<Self, u32> {
        occ::ALL1.iter().cloned().collect()
    }

    pub fn from_usize(x: usize) -> Option<Self> {
        use self::Occ::*;
        match x {
            0 => Some(I),
            1 => Some(A),
            _ => None,
        }
    }
}

pub mod occ {
    //! Convenient aliases for occupancies.
    use super::Occ;
    pub const ALL1: [Occ; 2] = [Occ::I, Occ::A];
    pub const ALL2: [[Occ; 2]; 4] = [II, AI, IA, AA];

    pub use super::Occ::I;
    pub use super::Occ::A;
    pub const II: [Occ; 2] = [Occ::I, Occ::I];
    pub const AI: [Occ; 2] = [Occ::A, Occ::I];
    pub const IA: [Occ; 2] = [Occ::I, Occ::A];
    pub const AA: [Occ; 2] = [Occ::A, Occ::A];
}

/// Occupancy of antisymmetrized two-particle states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Occ20 {
    II,
    AI,
    AA,
}

impl From<Occ20> for usize {
    fn from(x: Occ20) -> Self {
        use self::Occ20::*;
        match x {
            II => 0,
            AI => 1,
            AA => 2,
        }
    }
}

impl From<Occ20> for u32 {
    fn from(x: Occ20) -> Self {
        usize::from(x) as _
    }
}

impl Increment for Occ20 {
    fn increment(&self) -> Option<Self> {
        Self::from_usize(usize::from(*self) + 1)
    }
}

impl Occ20 {
    pub fn chart() -> HashChart<Self, u32> {
        use self::Occ20::*;
        [II, AI, AA].iter().cloned().collect()
    }

    pub fn from_usize(x: usize) -> Option<Self> {
        use self::Occ20::*;
        match x {
            0 => Some(II),
            1 => Some(AI),
            2 => Some(AA),
            _ => None,
        }
    }
}

pub trait IndexBlockVec {
    type Elem;
    fn index_block_vec(&self, l: usize, u: usize) -> &Self::Elem;
}

pub trait IndexBlockVecMut: IndexBlockVec {
    fn index_block_vec_mut(&mut self, l: usize, u: usize) -> &mut Self::Elem;
}

impl<T> IndexBlockVec for Vec<Vec<T>> {
    type Elem = T;
    fn index_block_vec(&self, l: usize, u: usize) -> &Self::Elem {
        &self[l][u]
    }
}

impl<T> IndexBlockVecMut for Vec<Vec<T>> {
    fn index_block_vec_mut(&mut self, l: usize, u: usize) -> &mut Self::Elem {
        &mut self[l][u]
    }
}

pub trait IndexBlockMat {
    type Elem;
    fn index_block_mat(
        &self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &Self::Elem;
}

pub trait IndexBlockMatMut: IndexBlockMat {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem;
}

impl<'a, T> IndexBlockMat for BlockMat<'a, T> {
    type Elem = T;
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<'a, T> IndexBlockMat for BlockMatMut<'a, T> {
    type Elem = T;
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self.as_ref().get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<'a, T> IndexBlockMatMut for BlockMatMut<'a, T> {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem {
        self.as_mut().get(l).unwrap().get(u1, u2).unwrap()
    }
}

impl<T> IndexBlockMat for Vec<Matrix<T>> {
    type Elem = T;
    fn index_block_mat(&self, l: usize, u1: usize, u2: usize) -> &Self::Elem {
        self[l].as_ref().get(u1, u2).unwrap()
    }
}

impl<T> IndexBlockMatMut for Vec<Matrix<T>> {
    fn index_block_mat_mut(
        &mut self,
        l: usize,
        u1: usize,
        u2: usize,
    ) -> &mut Self::Elem {
        self[l].as_mut().get(u1, u2).unwrap()
    }
}
