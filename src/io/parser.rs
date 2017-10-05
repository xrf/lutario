use std::{io, mem};
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
pub struct Buffer {
    buf: Box<[u8]>,
    len: usize,
}

impl Deref for Buffer {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.buf[.. self.len]
    }
}

impl Buffer {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: vec![0; cap].into_boxed_slice(),
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn read_from<R: io::Read>(&mut self, mut r: R) -> io::Result<()> {
        self.len += r.read(&mut self.buf[self.len ..])?;
        Ok(())
    }
}

#[derive(Debug)]
pub struct Chain {
    buf: Buffer,
    next: SharedChain,
}

#[derive(Clone, Debug)]
struct SharedChainInner {
    chain: Rc<RefCell<Chain>>,
    alloc: Alloc,
}

#[derive(Clone, Debug, Default)]
pub struct SharedChain(Option<SharedChainInner>);

impl Drop for SharedChain {
    fn drop(&mut self) {
        if let Some(SharedChainInner { chain, alloc }) =
            mem::replace(&mut self.0, None)
        {
            alloc.recycle(chain);
        }
    }
}

impl SharedChain {
    pub fn new(chain: Rc<RefCell<Chain>>, alloc: Alloc) -> Self {
        SharedChain(Some(SharedChainInner { chain, alloc }))
    }

    pub fn replace_buf(&self, buf: Buffer) -> Option<Buffer> {
        self.0.as_ref().map(|chain| {
            mem::replace(&mut chain.chain.borrow_mut().buf, buf)
        })
    }

    pub fn get_or_allocate_next(&self) -> Option<SharedChain> {
        self.0.as_ref().map(|chain| {
            let next = &mut chain.chain.borrow_mut().next;
            if next.0.is_none() {
                *next = chain.alloc.allocate();
            }
            next.clone()
        })
    }
}

#[derive(Debug)]
struct AllocInner {
    buf_cap: usize,
    // we keep one chain in reserve: even with small lookaheads we might
    // need two buffers when we are parsing near the boundary
    reserve: Option<Rc<RefCell<Chain>>>,
}

#[derive(Debug, Clone)]
pub struct Alloc(Rc<RefCell<AllocInner>>);

impl Alloc {
    pub fn new(buf_cap: usize) -> Self {
        Alloc(Rc::new(RefCell::new(AllocInner {
            buf_cap,
            reserve: None,
        })))
    }

    pub fn allocate(&self) -> SharedChain {
        let buf_cap = {
            let mut alloc = self.0.borrow_mut();
            if let Some(chain) = alloc.reserve.take() {
                return SharedChain::new(chain, self.clone());
            }
            alloc.buf_cap
        };
        SharedChain::new(
            Rc::new(RefCell::new(Chain {
                next: Default::default(),
                buf: Buffer::with_capacity(buf_cap),
            })),
            self.clone(),
        )
    }

    pub fn recycle(&self, mut chain: Rc<RefCell<Chain>>) {
        let cap = Rc::get_mut(&mut chain).map(|chain| {
            let chain = chain.get_mut();
            chain.next = Default::default();
            chain.buf.clear();
            chain.buf.capacity()
        });
        if let Some(cap) = cap {
            let mut alloc = self.0.borrow_mut();
            assert_eq!(alloc.buf_cap, cap,
                       "recycled chain is missing buffer");
            alloc.reserve = Some(chain);
        }
    }
}

#[derive(Debug)]
pub struct Parser<R> {
    reader: R,
    index: usize,
    // note: we spill the buf of chain into buf, leaving chain.buf empty
    buf: Buffer,
    chain: SharedChain,
}

impl<R> Drop for Parser<R> {
    fn drop(&mut self) {
        // put the buf back or else Drop for SharedChain will fail
        let buf = mem::replace(&mut self.buf, Default::default());
        self.chain.replace_buf(buf).unwrap();
    }
}

#[derive(Clone, Debug)]
pub struct State {
    index: usize,
    chain: SharedChain,
}

impl<R> Parser<R> {
    pub fn with_capacity(reader: R, buf_cap: usize) -> Self {
        let alloc = Alloc::new(buf_cap);
        let chain = alloc.allocate();
        Self {
            reader,
            index: 0,
            buf: chain.replace_buf(Default::default()).unwrap(),
            chain,
        }
    }

    pub fn save(&self) -> State {
        State {
            index: self.index,
            chain: self.chain.clone(),
        }
    }

    pub fn restore(&mut self, state: State) {
        self.index = state.index;
        let old_chain = mem::replace(&mut self.chain, state.chain);
        let new_buf = self.chain.replace_buf(Default::default()).unwrap();
        if new_buf.capacity() != 0 {
            // if capacity is zero, then this state shares the same buffer as
            // the old one
            let old_buf = mem::replace(&mut self.buf, new_buf);
            old_chain.replace_buf(old_buf).unwrap();
        }
    }
}

impl<R: io::Read> Parser<R> {
    pub fn refill(&mut self) -> io::Result<()> {
        if self.buf.len() >= self.buf.capacity() {
            let new_chain = self.chain.get_or_allocate_next().unwrap();
            let new_buf = new_chain.replace_buf(Default::default()).unwrap();
            let old_buf = mem::replace(&mut self.buf, new_buf);
            self.chain.replace_buf(old_buf).unwrap();
            self.chain = new_chain;
            self.index = 0;
            assert!(self.buf.capacity() > 0);
        }
        self.buf.read_from(&mut self.reader)?;
        Ok(())
    }

    pub fn get(&mut self) -> io::Result<Option<u8>> {
        if let Some(&c) = self.buf[self.index ..].first() {
            self.index += 1;
            Ok(Some(c))
        } else {
            self.refill()?;
            if let Some(&c) = self.buf[self.index ..].first() {
                self.index += 1;
                Ok(Some(c))
            } else {
                Ok(None)
            }
        }
    }

    pub fn match_pred<F>(&mut self, pred: F) -> io::Result<Option<u8>>
        where F: FnOnce(u8) -> bool
    {
        let state = self.save();
        match self.get()? {
            Some(c) if pred(c) => Ok(Some(c)),
            _ => {
                self.restore(state);
                Ok(None)
            }
        }
    }

    pub fn match_bytes(&mut self, s: &[u8]) -> io::Result<bool> {
        let state = self.save();
        for &sc in s {
            if !self.match_pred(|c| c == sc)?.is_some() {
                self.restore(state);
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[test]
fn test() {
    let mut p = Parser::with_capacity(b"abcDEFghiJKL" as &[_], 3);
    assert_eq!(p.get().unwrap(), Some(b'a'));
    assert_eq!(p.get().unwrap(), Some(b'b'));
    assert_eq!(p.get().unwrap(), Some(b'c'));
    assert_eq!(p.get().unwrap(), Some(b'D'));

    let s = p.save();
    assert_eq!(p.get().unwrap(), Some(b'E'));
    assert_eq!(p.get().unwrap(), Some(b'F'));

    p.restore(s.clone());
    assert_eq!(p.get().unwrap(), Some(b'E'));
    assert_eq!(p.get().unwrap(), Some(b'F'));

    p.restore(s.clone());
    assert_eq!(p.get().unwrap(), Some(b'E'));
    assert_eq!(p.get().unwrap(), Some(b'F'));
    assert_eq!(p.get().unwrap(), Some(b'g'));
    assert_eq!(p.get().unwrap(), Some(b'h'));

    p.restore(s.clone());
    assert_eq!(p.get().unwrap(), Some(b'E'));
    assert_eq!(p.get().unwrap(), Some(b'F'));
    assert_eq!(p.get().unwrap(), Some(b'g'));
    assert_eq!(p.get().unwrap(), Some(b'h'));
    assert_eq!(p.get().unwrap(), Some(b'i'));
    assert_eq!(p.get().unwrap(), Some(b'J'));

    p.restore(s);
    assert_eq!(p.get().unwrap(), Some(b'E'));
    assert_eq!(p.get().unwrap(), Some(b'F'));
    assert_eq!(p.get().unwrap(), Some(b'g'));
    assert_eq!(p.get().unwrap(), Some(b'h'));
    assert_eq!(p.get().unwrap(), Some(b'i'));
    assert_eq!(p.get().unwrap(), Some(b'J'));

    let t = p.save();
    assert_eq!(p.get().unwrap(), Some(b'K'));
    assert_eq!(p.get().unwrap(), Some(b'L'));
    assert_eq!(p.get().unwrap(), None);

    p.restore(t);
    assert_eq!(p.get().unwrap(), Some(b'K'));
    assert_eq!(p.get().unwrap(), Some(b'L'));
    assert_eq!(p.get().unwrap(), None);
}
