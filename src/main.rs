use std::io::{self, Write};

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


#[derive(Debug, Copy, Clone, ToString, PartialEq)]
pub enum Token {
    B, D, F, G, H, J, K, M, N, P, R, S, T, V, W, Y, Z,
    SH, TS, CH,
}

#[derive(Debug, Copy, Clone, ToString, PartialEq)]
pub enum Symbol {
    Init,
    X,
    Q,
    C,
    PH,
    TH,
    Simple(Token) // token is same as its pronunciation
}
// is_beggining_of_phoneme, symbol
type State = (bool, Symbol);

fn parse(english: &String) -> Vec<Phoneme> {
    let mut state: State = (true, Symbol::Init);
    let mut vec: Vec<Phoneme> = vec![Phoneme::Silence];
    let flush = |symbol: Symbol, vec: &mut Vec<Phoneme>| match symbol {
        Symbol::Init => (),
        Symbol::X => {vec.push(Phoneme::Conso("k")); vec.push(Phoneme::Vowel("u")); vec.push(Phoneme::Conso("s"))},
        Symbol::Q => vec.push(Phoneme::Conso("k")),
        Symbol::C => vec.push(Phoneme::Conso("k")),
        Symbol::TH => vec.push(Phoneme::Conso("s")),
        Symbol::PH => vec.push(Phoneme::Conso("f")),
        Symbol::Simple(tok) => vec.push(match tok {
            Token::B => Phoneme::Conso("b"),
            Token::D => Phoneme::Conso("d"),
            Token::F => Phoneme::Conso("f"),
            Token::G => Phoneme::Conso("g"),
            Token::H => Phoneme::Conso("h"),
            Token::J => Phoneme::Conso("j"),
            Token::K => Phoneme::Conso("k"),
            Token::M => Phoneme::Conso("m"),
            Token::N => Phoneme::Conso("n"),
            Token::P => Phoneme::Conso("p"),
            Token::R => Phoneme::Conso("r"),
            Token::S => Phoneme::Conso("s"),
            Token::T => Phoneme::Conso("t"),
            Token::V => Phoneme::Conso("v"),
            Token::W => Phoneme::Conso("w"),
            Token::Y => Phoneme::Conso("y"),
            Token::Z => Phoneme::Conso("z"),
            Token::SH => Phoneme::Conso("sh"),
            Token::TS => Phoneme::Conso("ts"),
            Token::CH => Phoneme::Conso("ch"),
        })
    };

    for c in english.to_lowercase().chars() {
        state = match (state, c) {
            // vowel input
            ((_, sym), 'a') => {flush(sym, &mut vec); vec.push(Phoneme::Vowel("a")); (false, Symbol::Init)},
            ((_, sym), 'i') => {flush(sym, &mut vec); vec.push(Phoneme::Vowel("i")); (false, Symbol::Init)},
            ((_, sym), 'u') => {flush(sym, &mut vec); vec.push(Phoneme::Vowel("u")); (false, Symbol::Init)},
            ((_, sym), 'e') => {flush(sym, &mut vec); vec.push(Phoneme::Vowel("e")); (false, Symbol::Init)},
            ((_, sym), 'o') => {flush(sym, &mut vec); vec.push(Phoneme::Vowel("o")); (false, Symbol::Init)},

            // 'y' as vowel
            ((bos, Symbol::Init), 'y') => (bos, Symbol::Simple(Token::Y)),
            ((_, other), 'y') => {flush(other, &mut vec); vec.push(Phoneme::Vowel("i")); vec.push(Phoneme::Vowel("i")); (false, Symbol::Init)},

            // stack consonants
            ((bos, Symbol::Simple(Token::S)), 'h') => (bos, Symbol::Simple(Token::SH)),
            ((bos, Symbol::Simple(Token::T)), 's') => (bos, Symbol::Simple(Token::TS)),
            ((bos, Symbol::Simple(Token::T)), 'h') => (bos, Symbol::TH),
            ((bos, Symbol::Simple(Token::P)), 'h') => (bos, Symbol::PH),
            ((bos, Symbol::C), 'h') => (bos, Symbol::Simple(Token::CH)),

            // consonant input
            ((bos, symbol), char) => {
                let mut out_bos = bos;
                match symbol {
                    Symbol::Init => (),
                    other => {
                        // flush prev consonant

                        if !bos && symbol == Symbol::Simple(Token::N) {
                            // N as vowel
                            vec.push(Phoneme::Vowel("N"));
                        } else {
                            // unvoice
                            flush(other, &mut vec);
                            vec.push(Phoneme::Vowel("u"));
                        }
                        out_bos = false;
                    }
                };
                (out_bos, match char {
                    'b' => Symbol::Simple(Token::B),
                    'c' => Symbol::C,
                    'd' => Symbol::Simple(Token::D),
                    'f' => Symbol::Simple(Token::F),
                    'g' => Symbol::Simple(Token::G),
                    'h' => Symbol::Simple(Token::H),
                    'j' => Symbol::Simple(Token::J),
                    'k' => Symbol::Simple(Token::K),
                    'l' => Symbol::Simple(Token::R),
                    'm' => Symbol::Simple(Token::M),
                    'n' => Symbol::Simple(Token::N),
                    'p' => Symbol::Simple(Token::P),
                    'q' => Symbol::Q,
                    'r' => Symbol::Simple(Token::R),
                    's' => Symbol::Simple(Token::S),
                    't' => Symbol::Simple(Token::T),
                    'v' => Symbol::Simple(Token::V),
                    'w' => Symbol::Simple(Token::W),
                    'x' => Symbol::X,
                    'y' => Symbol::Simple(Token::Y),
                    'z' => Symbol::Simple(Token::Z),
                    _ => Symbol::Init,
                })
            },
        }
    }
    vec
}

fn unwrap_phonemes<'a>(seq: &'a Vec<Phoneme>) -> Vec<&'a str> {
    seq.iter().map(|x| match x {
        Phoneme::Silence => "sil",
        Phoneme::Vowel(x) => x,
        Phoneme::Conso(x) => x,
    }).collect()
}

fn to_kana(seq: &Vec<Phoneme>) -> String {
    let mut out = String::new();
    for w in seq.windows(2) {
        match (w[0], w[1]) {
            (Phoneme::Silence, Phoneme::Vowel(x)) => out.push_str(vowel_to_kana(x)),
            (Phoneme::Vowel(_), Phoneme::Vowel(x)) => out.push_str(vowel_to_kana(x)),
            (Phoneme::Conso(x), Phoneme::Vowel(y)) => out.push_str(convo_to_kana(x, y)),
            _ => (),
        }
    }
    out
}

fn vowel_to_kana(x: &str) -> &str {
    match &*x.to_lowercase() {
        "a" => "ア",
        "i" => "イ",
        "u" => "ウ",
        "e" => "エ",
        "o" => "オ",
        "n" => "ン",
        "cl" => "ッ",
        _ => "",
    }
}

fn convo_to_kana<'a>(c: &'a str, v: &'a str) -> &'a str {
    let vv = &*v.to_lowercase();
    match c {
        "b"  => match vv {"a"=>"バ",	"i"=>"ビ",	"u"=>"ブ",	"e"=>"ベ",	"o"=>"ボ",	_=>""},
        "by" => match vv {"a"=>"ビャ",	"i"=>"ビ",	"u"=>"ビュ",	"e"=>"ビェ",	"o"=>"ビョ",	_=>""},
        "ch" => match vv {"a"=>"チャ",	"i"=>"チ",	"u"=>"チュ",	"e"=>"チェ",	"o"=>"チョ",	_=>""},
        "d"  => match vv {"a"=>"ダ",	"i"=>"ジ",	"u"=>"ズ",	"e"=>"デ",	"o"=>"ド",	_=>""},
        // "dy" => match vv {},
        "f"  => match vv {"a"=>"ファ",	"i"=>"フィ",	"u"=>"フ",	"e"=>"フェ",	"o"=>"フォ",	_=>""},
        "g"  => match vv {"a"=>"ガ",	"i"=>"ギ",	"u"=>"グ",	"e"=>"ゲ",	"o"=>"ゴ",	_=>""},
        // "gw" => match vv {"a"=>"",	"i"=>"",	"u"=>"",	"e"=>"",	"o"=>"",	_=>""},
        "gy" => match vv {"a"=>"ギャ",	"i"=>"ギ",	"u"=>"ギュ",	"e"=>"ギェ",	"o"=>"ギョ",	_=>""},
        "h"  => match vv {"a"=>"ハ",	"i"=>"ヒ",	"u"=>"フ",	"e"=>"ヘ",	"o"=>"ホ",	_=>""},
        "hy" => match vv {"a"=>"ヒャ",	"i"=>"ヒ",	"u"=>"ヒュ",	"e"=>"ヒェ",	"o"=>"ヒョ",	_=>""},
        "j"  => match vv {"a"=>"ジャ",	"i"=>"ジ",	"u"=>"ジュ",	"e"=>"ジェ",	"o"=>"ジョ",	_=>""},
        "k"  => match vv {"a"=>"カ",	"i"=>"キ",	"u"=>"ク",	"e"=>"ケ",	"o"=>"コ",	_=>""},
        // "kw" => match vv {"a"=>"",	"i"=>"",	"u"=>"",	"e"=>"",	"o"=>"",	_=>""},
        "ky" => match vv {"a"=>"キャ",	"i"=>"キ",	"u"=>"キュ",	"e"=>"キェ",	"o"=>"キョ",	_=>""},
        "m"  => match vv {"a"=>"マ",	"i"=>"ミ",	"u"=>"ム",	"e"=>"メ",	"o"=>"モ",	_=>""},
        "my" => match vv {"a"=>"ミャ",	"i"=>"ミ",	"u"=>"ミュ",	"e"=>"ミェ",	"o"=>"ミョ",	_=>""},
        "n"  => match vv {"a"=>"ナ",	"i"=>"ニ",	"u"=>"ヌ",	"e"=>"ネ",	"o"=>"ノ",	_=>""},
        "ny" => match vv {"a"=>"ニャ",	"i"=>"ニ",	"u"=>"ニュ",	"e"=>"ニェ",	"o"=>"ニョ",	_=>""},
        "p"  => match vv {"a"=>"パ",	"i"=>"ピ",	"u"=>"プ",	"e"=>"ペ",	"o"=>"ポ",	_=>""},
        "py" => match vv {"a"=>"ピャ",	"i"=>"ピ",	"u"=>"ピュ",	"e"=>"ピェ",	"o"=>"ピョ",	_=>""},
        "r"  => match vv {"a"=>"ラ",	"i"=>"リ",	"u"=>"ル",	"e"=>"レ",	"o"=>"ロ",	_=>""},
        "ry" => match vv {"a"=>"リャ",	"i"=>"リ",	"u"=>"リュ",	"e"=>"レ",	"o"=>"リョ",	_=>""},
        "s"  => match vv {"a"=>"サ",	"i"=>"シ",	"u"=>"ス",	"e"=>"セ",	"o"=>"ソ",	_=>""},
        "sh" => match vv {"a"=>"シャ",	"i"=>"シ",	"u"=>"シュ",	"e"=>"シェ",	"o"=>"ショ",	_=>""},
        "t"  => match vv {"a"=>"タ",	"i"=>"チ",	"u"=>"ツ",	"e"=>"テ",	"o"=>"ト",	_=>""},
        "ts" => match vv {"a"=>"ツァ",	"i"=>"チ",	"u"=>"ツ",	"e"=>"ツェ",	"o"=>"ツォ",	_=>""},
        "ty" => match vv {"a"=>"チャ",	"i"=>"チ",	"u"=>"チュ",	"e"=>"チェ",	"o"=>"チョ",	_=>""},
        "v"  => match vv {"a"=>"ヴァ",	"i"=>"ヴィ",	"u"=>"ヴ",	"e"=>"ヴェ",	"o"=>"ヴォ",	_=>""},
        "w"  => match vv {"a"=>"ワ",	"i"=>"ウィ",	"u"=>"ウ",	"e"=>"ウェ",	"o"=>"ウォ",	_=>""},
        "y"  => match vv {"a"=>"ヤ",	"i"=>"イ",	"u"=>"ユ",	"e"=>"エ",	"o"=>"ヨ",	_=>""},
        "z"  => match vv {"a"=>"ザ",	"i"=>"ジ",	"u"=>"ズ",	"e"=>"ゼ",	"o"=>"ゾ",	_=>""},
        _ => "",
    }
}

pub fn convert(english: &String) -> String {
    let fine = parse(&english);
    assert_eq!(ValidationResult::Ok, validate(&fine));
    // let result = unwrap_phonemes(&fine);

    // println!("Input: {}", english);
    // println!("Parsed: {:?}", fine);
    // println!("Result: {:?}", result);

    // result.join(" ")

    to_kana(&fine)
}

fn main() {
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut query = String::new();
        io::stdin().read_line(&mut query).expect("input error.");
        println!("{}", convert(&query));
    }
}
