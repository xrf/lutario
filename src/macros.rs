macro_rules! re {
    ($e:expr) => {
        {
            lazy_static! {
                static ref REGEX: Regex = Regex::new($e).unwrap();
            }
            &REGEX
        }
    }
}
