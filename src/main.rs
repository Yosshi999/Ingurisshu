use std::io::{self, Write};
use aho_corasick::{AhoCorasickBuilder, MatchKind};

extern crate strum;
#[macro_use] extern crate strum_macros;
use strum::{IntoEnumIterator};

pub const OPENJTALK_VOWELS: [&str; 12] = [
    "a", "i", "u", "e", "o", "A", "I", "U", "E", "O", "N", "cl",
];
pub const OPENJTALK_CONSONANTS: [&str; 32] = [
    "b", "by", "ch", "d", "dy", "f", "g", "gw", "gy", "h", "hy", "j", "k", "kw", "ky",
    "m", "my", "n", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts", "ty", "v", "w", "y", "z"
];

#[derive(Debug, Copy, Clone)]
pub enum Phoneme<'a> {
    Silence,
    Vowel(&'a str),
    Conso(&'a str)
}

#[derive(Debug, PartialEq)]
pub enum ValidationResult {
    Ok,
    UnknownPhoneme(usize),
}

pub fn validate(phonemes: &Vec<Phoneme>) -> ValidationResult {
    for i in 0..phonemes.len() {
        let valid = match &phonemes[i] {
            Phoneme::Silence => true,
            Phoneme::Vowel(x) => OPENJTALK_VOWELS.contains(x),
            Phoneme::Conso(x) => OPENJTALK_CONSONANTS.contains(x)
        };
        if !valid {
            return ValidationResult::UnknownPhoneme(i);
        }
    };
    ValidationResult::Ok
}

#[derive(Debug, Copy, Clone, ToString, EnumIter)]
pub enum Token {
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    CH, GW, KW, SH, TS, TH, PH
}

fn tokenize(english: &String) -> Vec<Token> {
    let tokens: Vec<Token> = Token::iter().collect();
    let patterns_string: Vec<String> = tokens.iter().map(|x| x.to_string()).collect();
    let patterns: Vec<&str> = patterns_string.iter().map(|x| &**x).collect();

    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(true)
        .match_kind(MatchKind::LeftmostLongest)
        .build(&patterns);

    ac.find_iter(english).map(|mat| tokens[mat.pattern()]).collect()
}

fn coarse_phonemes(seq: &Vec<Token>) -> Vec<Phoneme> {
    let mut vec = vec![vec![Phoneme::Silence]];
    for t in seq {
        vec.push(match t {
            Token::A => vec![Phoneme::Vowel("a")],
            Token::I => vec![Phoneme::Vowel("i")],
            Token::U => vec![Phoneme::Vowel("u")],
            Token::E => vec![Phoneme::Vowel("e")],
            Token::O => vec![Phoneme::Vowel("o")],

            Token::B => vec![Phoneme::Conso("b")],
            Token::C => vec![Phoneme::Conso("s")],
            Token::D => vec![Phoneme::Conso("d")],
            Token::F => vec![Phoneme::Conso("f")],
            Token::G => vec![Phoneme::Conso("g")],
            Token::H => vec![Phoneme::Conso("h")],
            Token::J => vec![Phoneme::Conso("j")],
            Token::K => vec![Phoneme::Conso("k")],
            Token::L => vec![Phoneme::Conso("r")],
            Token::M => vec![Phoneme::Conso("m")],
            Token::N => vec![Phoneme::Conso("n")],
            Token::P => vec![Phoneme::Conso("p")],
            Token::Q => vec![Phoneme::Conso("k")],
            Token::R => vec![Phoneme::Conso("r")],
            Token::S => vec![Phoneme::Conso("s")],
            Token::T => vec![Phoneme::Conso("t")],
            Token::V => vec![Phoneme::Conso("v")],
            Token::W => vec![Phoneme::Conso("w")],
            Token::X => vec![Phoneme::Conso("k"), Phoneme::Conso("s")],
            Token::Y => vec![Phoneme::Conso("y")],
            Token::Z => vec![Phoneme::Conso("z")],

            Token::CH => vec![Phoneme::Conso("ch")],
            Token::GW => vec![Phoneme::Conso("gw")],
            Token::KW => vec![Phoneme::Conso("kw")],
            Token::SH => vec![Phoneme::Conso("sh")],
            Token::TS => vec![Phoneme::Conso("ts")],
            Token::TH => vec![Phoneme::Conso("s")],
            Token::PH => vec![Phoneme::Conso("f")],
        });
    }
    vec.push(vec![Phoneme::Silence]);
    vec.into_iter().flatten().collect()
}

fn fine_phonemes<'a>(seq: &'a Vec<Phoneme>) -> Vec<Phoneme<'a>> {
    let mut vec = vec![Phoneme::Silence];
    for w in seq.windows(3) {
        match (w[0], w[1], w[2]) {
            (Phoneme::Silence, Phoneme::Conso("n"), Phoneme::Conso(_)) => (),
            (Phoneme::Silence, x, _) => vec.push(x),
            (Phoneme::Conso(_), Phoneme::Vowel(x), _) => vec.push(Phoneme::Vowel(x)),
            (Phoneme::Vowel(_), Phoneme::Vowel(x), _) => vec.push(Phoneme::Vowel(x)),
            (Phoneme::Conso("y"), x, _) => vec.push(x),
            (Phoneme::Conso(_), Phoneme::Conso("y"), _) => {vec.push(Phoneme::Vowel("i")); vec.push(Phoneme::Vowel("i"))},

            (Phoneme::Vowel(_), Phoneme::Conso("n"), Phoneme::Conso(_)) => vec.push(Phoneme::Vowel("N")),
            (Phoneme::Conso("n"), Phoneme::Conso("n"), Phoneme::Conso(_)) => vec.push(Phoneme::Vowel("N")),
            (Phoneme::Conso(_), Phoneme::Conso("n"), Phoneme::Conso(_)) => {vec.push(Phoneme::Vowel("u")); vec.push(Phoneme::Vowel("N"))},
            (Phoneme::Vowel(_), Phoneme::Conso("n"), Phoneme::Silence) => vec.push(Phoneme::Vowel("N")),
            (Phoneme::Conso(_), Phoneme::Conso("n"), Phoneme::Silence) => {vec.push(Phoneme::Vowel("u")); vec.push(Phoneme::Vowel("N"))},
            (Phoneme::Conso("n"), Phoneme::Conso(x), _) => vec.push(Phoneme::Conso(x)),
            (Phoneme::Conso(_), Phoneme::Conso(x), _) => {vec.push(Phoneme::Vowel("U")); vec.push(Phoneme::Conso(x))},
            (_, Phoneme::Conso(x), Phoneme::Silence) => {vec.push(Phoneme::Conso(x)); vec.push(Phoneme::Vowel("U"))},
            (_, x, _) => vec.push(x)
        };
    }
    vec.push(Phoneme::Silence);
    vec
}

fn unwrap_phonemes<'a>(seq: &'a Vec<Phoneme>) -> Vec<&'a str> {
    seq.iter().map(|x| match x {
        Phoneme::Silence => "sil",
        Phoneme::Vowel(x) => x,
        Phoneme::Conso(x) => x,
    }).collect()
}

pub fn convert(english: &String) -> String {
    let matches = tokenize(&english);
    let coarse = coarse_phonemes(&matches);
    assert_eq!(ValidationResult::Ok, validate(&coarse));
    let fine = fine_phonemes(&coarse);
    assert_eq!(ValidationResult::Ok, validate(&fine));
    let result = unwrap_phonemes(&fine);
    
    // println!("Input: {}", english);
    // println!("Tokens: {:?}", matches);
    // println!("Step1: {:?}", coarse);
    // println!("Step2: {:?}", fine);
    // println!("Result: {:?}", result);

    result.join(" ")
}

fn main() {
    print!("> ");
    io::stdout().flush().unwrap();
    let mut query = String::new();
    io::stdin().read_line(&mut query).expect("input error.");
    println!("{}", convert(&query));
}
