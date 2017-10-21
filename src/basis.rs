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
use std::fmt;
use std::borrow::Borrow;
use std::hash::Hash;
use std::ops::{Sub};
use std::sync::Arc;
use fnv::FnvHashMap;
use num::Zero;
use super::block_matrix::{BlockMat, BlockMatMut};
use super::matrix::{Mat, MatShape};
use super::utils;

/// An Abelian group.
pub trait Abelian: Zero + Sub<Output = Self> {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct State2<T>(pub T, pub T);

impl<T: fmt::Display> fmt::Display for State2<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {}", self.0, self.1)
    }
}

/// A chart is a bijection between a zero-based interval (i.e. indices) and an
/// arbitrary set of objects of type `T`.  `HashChart` is an implementation of
/// this idea using `HashMap`.
#[derive(Clone)]
pub struct HashChart<T> {
    pub encoder: FnvHashMap<T, usize>,
    pub decoder: Vec<T>,
}

impl<T> Default for HashChart<T> {
    fn default() -> Self {
        Self {
            encoder: utils::default_hash_map(),
            decoder: Default::default(),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for HashChart<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("HashChart").field(&self.decoder).finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HashChartInsertResult {
    /// Whether the insert was successful
    pub inserted: bool,
    /// Index of the inserted item
    pub index: usize,
}

impl<T> HashChart<T> {
    pub fn len(&self) -> usize {
        self.decoder.len()
    }

    pub fn decode(&self, i: usize) -> Option<&T> {
        self.decoder.get(i)
    }
}

impl<T: Hash + Eq> HashChart<T> {
    pub fn encode<Q>(&self, q: &Q) -> Option<usize>
        where Q: Hash + Eq,
              T: Borrow<Q>,
    {
        self.encoder.get(q).cloned()
    }
}

impl<T: Hash + Eq + Clone> HashChart<T> {
    pub fn insert(&mut self, t: T) -> HashChartInsertResult {
        let mut inserted = false;
        let query = t.clone();
        let decoder = &mut self.decoder;
        let index = *self.encoder.entry(query).or_insert_with(|| {
            inserted = true;
            let index = decoder.len();
            decoder.push(t);
            index
        });
        HashChartInsertResult { inserted, index }
    }
}

/// A basis layout has only enough information to understand the structure of
/// a matrix along one axis, but not to interpret its elements.
#[derive(Clone, Debug, Default)]
pub struct BasisLayout {
    pub num_parts: usize,
    /// `part_offsets[l * num_parts + x - 1] == ∑[x' < x] num_states_l_x(l, x')
    /// where l < nl && x > 0 && x ≤ num_parts`
    pub part_offsets: Vec<usize>,
    /// `chan_offsets[l] == ∑[l' < l] num_states_l(l')
    /// where l ≤ nl`
    pub chan_offsets: Vec<usize>,
}

impl BasisLayout {
    pub fn num_chans(&self) -> usize {
        self.chan_offsets.len() - 1
    }

    pub fn chan_offset(&self, l: usize) -> usize {
        self.chan_offsets[l]
    }

    pub fn part_offsets(&self) -> Mat<usize> {
        Mat::new(
            &mut (&self.part_offsets as _),
            MatShape::packed(
                self.part_offsets.len() / self.num_parts,
                self.num_parts,
            ).validate().unwrap(),
        ).unwrap()
    }

    pub fn chan_dim(&self, l: usize) -> usize {
        let n = self.chan_offset(l + 1) - self.chan_offset(l);
        debug_assert_eq!(self.part_offset(l, self.num_parts), n);
        n
    }

    pub fn part_offset(&self, l: usize, x: usize) -> usize {
        if x == 0 {
            0
        } else {
            *self.part_offsets().get(l, x - 1).unwrap()
        }
    }

    pub fn part_dim(&self, l: usize, x: usize) -> usize {
        self.part_offset(l, x + 1) - self.part_offset(l, x)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct State<L, X, U> {
    pub chan: L,
    pub part: X,
    pub aux: U,
}

/// A basis chart has all the information to interpret an axis of a
/// block-diagonal matrix.
#[derive(Clone)]
pub struct BasisChart<L, X, U> {
    pub chan_chart: HashChart<L>,
    pub part_chart: HashChart<X>,
    /// `aux_encoder[(l, U)] == u`
    pub aux_encoder: FnvHashMap<(usize, U), usize>,
    /// `aux_decoder[chan_offset(l) + u] == (L, U)`
    pub aux_decoder: Vec<U>,
    pub layout: Arc<BasisLayout>,
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

impl<L, X, U> BasisChart<L, X, U> {
    pub fn layout(&self) -> &Arc<BasisLayout> {
        &self.layout
    }

    pub fn decode_chan(&self, l: usize) -> Option<&L> {
        self.chan_chart.decode(l)
    }

    pub fn decode_part(&self, x: usize) -> Option<&X> {
        self.part_chart.decode(x)
    }

    pub fn decode_aux(&self, l: usize, u: usize) -> Option<&U> {
        self.aux_decoder.get(self.layout.chan_offset(l) + u)
    }
}

impl<L: Hash + Eq, X, U> BasisChart<L, X, U> {
    pub fn encode_chan(&self, l: &L) -> Option<usize> {
        self.chan_chart.encode(l)
    }
}

impl<L, X: Hash + Eq, U> BasisChart<L, X, U> {
    pub fn encode_part(&self, x: &X) -> Option<usize> {
        self.part_chart.encode(x)
    }
}

impl<L, X, U: Hash + Eq> BasisChart<L, X, U> {
    pub fn encode_aux(&self, l: usize, u: &U) -> Option<usize> {
        utils::with_tuple2_ref(&l, u, |lu| self.aux_encoder.get(lu).cloned())
    }
}

impl<L, X, U> BasisChart<L, X, U>
    where L: Hash + Eq + Clone,
          X: Hash + Eq + Clone,
          U: Hash + Eq + Clone,
{
    pub fn new(states: &mut Iterator<Item = State<L, X, U>>) -> Self {
        let mut chan_chart = HashChart::default();
        let mut part_chart = HashChart::default();
        let mut state_chart = HashChart::default(); // this one is temporary
        let mut lxps = Vec::default();
        for State { chan, part, aux } in states {
            let l = chan_chart.insert(chan.clone()).index;
            let x = part_chart.insert(part).index;
            let inserted_lu = state_chart.insert((l, aux));
            assert!(inserted_lu.inserted, "every state (l, u) must be unique");
            let p = inserted_lu.index;
            lxps.push((l, x, p));
        }
        lxps.sort_unstable();
        let nl = chan_chart.len();
        let nx = part_chart.len();
        let mut layout = BasisLayout {
            num_parts: nx,
            part_offsets: Default::default(),
            chan_offsets: vec![0],
        };
        let mut aux_decoder = Vec::default();
        let mut aux_encoder = FnvHashMap::default();
        let mut lxps = lxps.iter();
        for l in 0 .. nl {
            let mut u = 0;
            for x in 0 .. nx {
                loop {
                    let mut lxps_i = lxps.clone();
                    match lxps_i.next() {
                        Some(&(l_i, x_i, p_i)) if l == l_i && x == x_i => {
                            lxps = lxps_i;
                            let state = state_chart.decode(p_i).unwrap().clone();
                            aux_decoder.push(state.1.clone());
                            aux_encoder.insert(state, u);
                            u += 1;
                        }
                        _ => break,
                    }
                }
                layout.part_offsets.push(u);
            }
            layout.chan_offsets.push(aux_decoder.len());
        }
        Self {
            chan_chart,
            part_chart,
            aux_encoder,
            aux_decoder,
            layout: Arc::new(layout),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatLayout {
    pub block_offsets: Vec<usize>,
    pub block_strides: Arc<BasisLayout>,
    pub left: Arc<BasisLayout>,
    pub right: Arc<BasisLayout>,
}

impl MatLayout {
    pub fn new(left: Arc<BasisLayout>, right: Arc<BasisLayout>) -> Self {
        let mut block_offsets = vec![0];
        let mut i = 0;
        assert_eq!(left.num_chans(), right.num_chans());
        for l in 0 .. left.num_chans() {
            i += left.chan_dim(l) * right.chan_dim(l);
            block_offsets.push(i);
        }
        MatLayout {
            block_offsets,
            block_strides: right.clone(),
            left,
            right,
        }
    }

    pub fn len(&self) -> usize {
        *self.block_offsets.last().unwrap()
    }

    pub fn block_dim(&self, l: usize) -> usize {
        self.block_offsets[l + 1] - self.block_offsets[l]
        *self.block_offsets.last().unwrap()
    }

    pub fn offset(&self, l: usize, u1: usize, u2: usize) -> usize {
        debug_assert!(l < self.block_offsets.len() - 1);
        debug_assert!(u1 < self.left.chan_dim(l));
        debug_assert!(u2 < self.right.chan_dim(l));
        self.block_offsets[l] + u1 * self.block_strides.chan_dim(l) + u2
    }
}

#[derive(Debug)]
pub struct MatChart<'a, L: 'a, X1: 'a, X2: 'a, U1: 'a, U2: 'a> {
    pub layout: &'a MatLayout,
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
        let il = self.left.encode_chan(&l).expect("invalid channel");
        let iu1 = self.left.encode_aux(il, &u1).expect("invalid left state");
        let iu2 = self.right.encode_aux(il, &u2).expect("invalid right state");
        self.layout.offset(il, iu1, iu2)
    }
}

/// Orbital index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Orb(pub u32);

/// Channelized state index.
#[derive(Clone, Copy, Debug)]
pub struct ChanState {
    /// Channel index
    pub l: u32,
    /// Auxiliary index
    pub u: u32,
}

/// Occupancy of a state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Occ {
    /// Occupied ("hole") state
    I,
    /// Unoccupied ("particle") state
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

pub mod occ {
    //! Convenient aliases for occupancies.
    use super::Occ;

    pub use super::Occ::I;
    pub use super::Occ::A;
    pub const II: [Occ; 2] = [Occ::I, Occ::I];
    pub const AI: [Occ; 2] = [Occ::A, Occ::I];
    pub const IA: [Occ; 2] = [Occ::I, Occ::A];
    pub const AA: [Occ; 2] = [Occ::A, Occ::A];
}

/// Occupancy of antisymmetrized two-particle states.
#[derive(Clone, Copy, Debug)]
pub enum Occ20 {
    II,
    AI,
    AA,
}

impl Occ20 {
    pub fn from_usize(x: usize) -> Option<Self> {
        use self::Occ20::*;
        match x {
            0 => Some(II),
            1 => Some(AI),
            2 => Some(AA),
            _ => None,
        }
    }

    pub fn step(self) -> Option<Self> {
        Self::from_usize(usize::from(self) + 1)
    }
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
