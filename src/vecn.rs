//! A tiny module for tiny fixed-length vectors.
use std::ops::{Add, Index, IndexMut, Mul, Sub};
use super::utils::cast;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vec3<T: Copy>(pub [T; 3]);

impl<T: Copy> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vec3([x, y, z])
    }

    pub fn map<F, U>(self, mut f: F) -> Vec3<U>
        where F: FnMut(T) -> U,
              U: Copy
    {
        let y0 = f(self[0]);
        let y1 = f(self[1]);
        let y2 = f(self[2]);
        Vec3::new(y0, y1, y2)
    }
}

impl<T: Copy> Index<usize> for Vec3<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T: Copy> IndexMut<usize> for Vec3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Copy + Add<Output=T> + Mul<Output=T>> Vec3<T> {
    pub fn dot(self, other: Self) -> T {
        self[0] * other[0] +
        self[1] * other[1] +
        self[2] * other[2]
    }

    pub fn norm_sq(self) -> T {
        self.dot(self)
    }
}

impl<T: Copy + Add<Output=T>> Add for Vec3<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Vec3::new(
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
        )
    }
}

impl<T: Copy + Sub<Output=T>> Sub for Vec3<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Vec3::new(
            self[0] - other[0],
            self[1] - other[1],
            self[2] - other[2],
        )
    }
}

/// Specialization for 3D vectors of `i8` to avoid overflow problems.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vec3I8(pub Vec3<i8>);

// TODO: could Add and Sub overflow .. ?
impl Add for Vec3I8 {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Vec3I8(self.0 + other.0)
    }
}

impl Sub for Vec3I8 {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Vec3I8(self.0 - other.0)
    }
}

impl Vec3I8 {
    pub fn new(x: i64, y: i64, z: i64) -> Self {
        Vec3I8(Vec3::new(cast(x), cast(y), cast(z)))
    }

    pub fn norm_sq(self) -> i64 {
        // convert first to avoid overflowing i8!
        self.0.map(|x| x as _).norm_sq()
    }
}
