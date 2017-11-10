//! Input and output utility.
use std::{io, str};
use std::error::Error;
use std::fs::File;
use std::marker::PhantomData;
use std::path::{self, Path};
use byteorder::{ByteOrder, ReadBytesExt};

pub mod parser;
pub use self::parser::Parser;

/// Helper function for creating `io::Error` with
/// `io::ErrorKind::InvalidData`.
pub fn invalid_data<E: Into<Box<Error + Send + Sync>>>(error: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error)
}

pub fn fill_buf_with_retry<R>(r: &mut R) -> io::Result<&[u8]>
    where R: io::BufRead + ?Sized,
{
    loop {
        match r.fill_buf() {
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => (),
            _ => break,
        }
    }
    r.fill_buf()
}

pub fn copy_while<R, P, W>(r: &mut R, mut pred: P, w: &mut W)
                           -> io::Result<usize>
    where R: io::BufRead + ?Sized,
          P: FnMut(u8) -> bool,
          W: io::Write + ?Sized,
{
    // adapted from: https://doc.rust-lang.org/src/std/io/mod.rs.html#1310-1571
    let mut read = 0;
    loop {
        let (done, used) = {
            let available = fill_buf_with_retry(r)?;
            match available.iter().position(|&c| !pred(c)) {
                Some(i) => {
                    w.write(&available[..i + 1])?;
                    (true, i + 1)
                }
                None => {
                    w.write(available)?;
                    (false, available.len())
                }
            }
        };
        r.consume(used);
        read += used;
        if done || used == 0 {
            return Ok(read);
        }
    }
}

/// Works just like Python's `os.path.splitext`.  Note that the returned
/// extension includes the dot.  If there is no extension, returns an empty
/// string as the extension.  Only works on UTF-8 strings due to limitations
/// of the `std::path::Path` API.
pub fn split_extension(path: &Path) -> io::Result<(&str, &str)> {
    let path = path.to_str()
        .ok_or_else(|| invalid_data("path is not UTF-8"))?;
    match path.rfind('.') {
        None => Ok((path, "")),
        Some(i) => {
            let ext = &path[i ..];
            if ext.chars().any(|c| path::is_separator(c)) {
                Ok((path, ""))
            } else {
                Ok((&path[.. i], ext))
            }
        }
    }
}

/// Open a compressed file and decode based on the file extension.
/// If the extension does not end in "z", the file is read as-is.
pub fn open_compressed(path: &Path) -> io::Result<(&Path, Box<io::Read>)> {
    use flate2;
    use xz2;

    let file = File::open(path)?;
    let (rest, ext) = split_extension(path)?;
    let (rest, ext) = if ext.ends_with("z") {
        (rest.as_ref(), ext)
    } else {
        (path, "")
    };
    Ok((rest, match ext {
        "" => Box::new(file),
        ".gz" => Box::new(flate2::read::GzDecoder::new(file)?),
        ".xz" => Box::new(xz2::read::XzDecoder::new(file)),
        _ => return Err(invalid_data(
            format!("unrecognized compression format: {}", ext),
        )),
    }))
}

/// Parse numbers from a white-space separated text file.
/// Comments delimited by `(* … *)` are ignored.
pub struct MapleTableParser<R> {
    parser: Parser<R>,
    buf: Vec<u8>,
    status: Option<io::Error>,
}

impl<R> MapleTableParser<R> {
    pub fn new(r: R) -> Self {
        Self {
            parser: Parser::new(r),
            buf: Default::default(),
            status: None,
        }
    }

    pub fn status(self) -> Option<io::Error> {
        self.status
    }
}

impl<R: io::Read> MapleTableParser<R> {
    pub fn match_f64(&mut self) -> io::Result<Option<f64>> {
        // skip whitespace and comments
        loop {
            if self.parser.munch_whitespace()? {
            } else if self.parser.match_bytes(b"(*")? {
                while !self.parser.match_bytes(b"*)")? {
                    if self.parser.get()?.is_none() {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "incomplete comment",
                        ));
                    }
                }
            } else {
                break;
            }
        }

        // read a token
        self.buf.clear();
        while let Some(c) = self.parser.match_pred(|c| {
            !(c as char).is_whitespace() && c != b'('
        })? {
            self.buf.push(c);
        }
        if self.buf.is_empty() {
            return Ok(None);
        }

        // parse the token
        Ok(Some(
            str::from_utf8(&self.buf)
                .map_err(invalid_data)?
                .parse()
                .map_err(invalid_data)?
        ))
    }
}

impl<R: io::Read> Iterator for MapleTableParser<R> {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        match self.match_f64() {
            Err(e) => {
                self.status = Some(e);
                None
            }
            Ok(r) => r,
        }
    }
}

pub trait ReadBinFrom: Sized {
    fn read_bin_from<E: ByteOrder, R: io::Read>(r: R) -> io::Result<Self>;
}

impl ReadBinFrom for f32 {
    fn read_bin_from<E: ByteOrder, R: io::Read>(mut r: R) -> io::Result<Self> {
        r.read_f32::<E>()
    }
}

impl ReadBinFrom for f64 {
    fn read_bin_from<E: ByteOrder, R: io::Read>(mut r: R) -> io::Result<Self> {
        r.read_f64::<E>()
    }
}

pub struct BinArrayParser<T, E, R> {
    reader: io::BufReader<R>,
    status: Option<io::Error>,
    _phantom: PhantomData<(T, E)>,
}

impl<T, E, R> BinArrayParser<T, E, R> {
    pub fn status(self) -> Option<io::Error> {
        self.status
    }
}

impl<T, E, R: io::Read> BinArrayParser<T, E, R> {
    pub fn new(r: R) -> Self {
        Self {
            reader: io::BufReader::new(r),
            status: None,
            _phantom: PhantomData,
        }
    }
}

impl<T, E, R> Iterator for BinArrayParser<T, E, R>
    where T: ReadBinFrom,
          E: ByteOrder,
          R: io::Read,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        match Self::Item::read_bin_from::<E, _>(&mut self.reader) {
            Err(e) => {
                if e.kind() != io::ErrorKind::UnexpectedEof {
                    self.status = Some(e);
                }
                None
            }
            Ok(x) => Some(x),
        }
    }
}
