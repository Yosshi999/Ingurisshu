use std::{io::{self, Write}};

extern crate strum;
#[macro_use] extern crate strum_macros;

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
pub enum Symbol {
    A, I, U, E, O,
    AR, IR, UR, ER, OR,
    B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S, T, V, W, Y, Z,
    X,
    SH, TS, CH,
    PH, TH,
}

type State = Option<Symbol>;

fn parse(english: &String) -> Vec<Phoneme> {
    let mut state : State = None;
    let mut vec: Vec<Phoneme> = vec![Phoneme::Silence];
    let mut is_first_mora = true;
    let vowels = ['a', 'i', 'u', 'e', 'o', 'y'];
    let flush = |vec: &mut Vec<Phoneme>, conso: &'static str, voicy: bool| {
        vec.push(Phoneme::Conso(conso));
        if !voicy {
            vec.push(Phoneme::Vowel("u"));
        }
    };

    for c in english.to_lowercase().chars() {
        let new_state: State = match state {
            None => None,
            Some(x) => match (x, c) {
                (Symbol::A, 'r') => Some(Symbol::AR),
                (Symbol::I, 'r') => Some(Symbol::IR),
                (Symbol::U, 'r') => Some(Symbol::UR),
                (Symbol::E, 'r') => Some(Symbol::ER),
                (Symbol::O, 'r') => Some(Symbol::OR),
                (Symbol::S, 'h') => Some(Symbol::SH),
                (Symbol::T, 's') => Some(Symbol::TS),
                (Symbol::C, 'h') => Some(Symbol::CH),
                (Symbol::P, 'h') => Some(Symbol::PH),
                (Symbol::T, 'h') => Some(Symbol::TH),
                _ => None,
            }
        };

        if let Some(_) = new_state {
            state = new_state;
        } else {
            // flush
            match state {
                None => (),
                Some(x) => {
                    let voicy = vowels.contains(&c);
                    match x {
                        Symbol::A => {vec.push(Phoneme::Vowel("a"))}
                        Symbol::I => {vec.push(Phoneme::Vowel("i"))}
                        Symbol::U => {vec.push(Phoneme::Vowel("u"))}
                        Symbol::E => {vec.push(Phoneme::Vowel("e"))}
                        Symbol::O => {vec.push(Phoneme::Vowel("o"))}
                        Symbol::AR => {
                            if voicy {
                                vec.push(Phoneme::Vowel("a"));
                                vec.push(Phoneme::Conso("r"));
                            } else {
                                // long vowel
                                vec.push(Phoneme::Vowel("a"));
                                vec.push(Phoneme::Vowel("a"));
                            }
                        },
                        Symbol::IR => {
                            if voicy {
                                vec.push(Phoneme::Vowel("i"));
                                vec.push(Phoneme::Conso("r"));
                            } else {
                                // long vowel
                                vec.push(Phoneme::Vowel("a"));
                                vec.push(Phoneme::Vowel("a"));
                            }
                        },
                        Symbol::UR => {
                            if voicy {
                                vec.push(Phoneme::Vowel("u"));
                                vec.push(Phoneme::Conso("r"));
                            } else {
                                // long vowel
                                vec.push(Phoneme::Vowel("a"));
                                vec.push(Phoneme::Vowel("a"));
                            }
                        },
                        Symbol::ER => {
                            if voicy {
                                vec.push(Phoneme::Vowel("e"));
                                vec.push(Phoneme::Conso("r"));
                            } else {
                                // long vowel
                                vec.push(Phoneme::Vowel("a"));
                                vec.push(Phoneme::Vowel("a"));
                            }
                        },
                        Symbol::OR => {
                            if voicy {
                                vec.push(Phoneme::Vowel("o"));
                                vec.push(Phoneme::Conso("r"));
                            } else {
                                // long vowel
                                vec.push(Phoneme::Vowel("o"));
                                vec.push(Phoneme::Vowel("o"));
                            }
                        },
                        Symbol::Y => {
                            if voicy {
                                if c == 'y' {
                                    vec.push(Phoneme::Vowel("i"));
                                } else {
                                    vec.push(Phoneme::Conso("y"));
                                }
                            } else {
                                vec.push(Phoneme::Vowel("i"));
                                if !is_first_mora {
                                    vec.push(Phoneme::Vowel("i"));
                                }
                            }
                        },
                        Symbol::X => {
                            vec.push(Phoneme::Conso("k"));
                            vec.push(Phoneme::Vowel("u"));
                            flush(&mut vec, "s", voicy);
                        },
                        Symbol::Q => flush(&mut vec, "k", voicy),
                        Symbol::C => flush(&mut vec, "k", voicy),
                        Symbol::TH => flush(&mut vec, "s", voicy),
                        Symbol::PH => flush(&mut vec, "f", voicy),
                        Symbol::B => flush(&mut vec, "b", voicy),
                        Symbol::D => flush(&mut vec, "d", voicy),
                        Symbol::F => flush(&mut vec, "f", voicy),
                        Symbol::G => flush(&mut vec, "g", voicy),
                        Symbol::H => flush(&mut vec, "h", voicy),
                        Symbol::J => flush(&mut vec, "j", voicy),
                        Symbol::K => flush(&mut vec, "k", voicy),
                        Symbol::L => flush(&mut vec, "r", voicy),
                        Symbol::M => flush(&mut vec, "m", voicy),
                        Symbol::N => {
                            if vowels.contains(&c) {
                                vec.push(Phoneme::Conso("n"));
                            } else if is_first_mora {
                                vec.push(Phoneme::Conso("n"));
                                vec.push(Phoneme::Vowel("u"));
                            } else {
                                vec.push(Phoneme::Vowel("N"));
                            }
                        },
                        Symbol::P => flush(&mut vec, "p", voicy),
                        Symbol::R => flush(&mut vec, "r", voicy),
                        Symbol::S => flush(&mut vec, "s", voicy),
                        Symbol::T => flush(&mut vec, "t", voicy),
                        Symbol::V => flush(&mut vec, "v", voicy),
                        Symbol::W => flush(&mut vec, "w", voicy),
                        Symbol::Z => flush(&mut vec, "z", voicy),
                        Symbol::SH => flush(&mut vec,"sh", voicy),
                        Symbol::TS => flush(&mut vec,"ts", voicy),
                        Symbol::CH => flush(&mut vec,"ch", voicy),
                    };
                    is_first_mora = false;
                }
            };

            // update state
            state = match c {
                'a' => Some(Symbol::A),
                'b' => Some(Symbol::B),
                'c' => Some(Symbol::C),
                'd' => Some(Symbol::D),
                'e' => Some(Symbol::E),
                'f' => Some(Symbol::F),
                'g' => Some(Symbol::G),
                'h' => Some(Symbol::H),
                'i' => Some(Symbol::I),
                'j' => Some(Symbol::J),
                'k' => Some(Symbol::K),
                'l' => Some(Symbol::L),
                'm' => Some(Symbol::M),
                'n' => Some(Symbol::N),
                'o' => Some(Symbol::O),
                'p' => Some(Symbol::P),
                'q' => Some(Symbol::Q),
                'r' => Some(Symbol::R),
                's' => Some(Symbol::S),
                't' => Some(Symbol::T),
                'u' => Some(Symbol::U),
                'v' => Some(Symbol::V),
                'w' => Some(Symbol::W),
                'x' => Some(Symbol::X),
                'y' => Some(Symbol::Y),
                'z' => Some(Symbol::Z),
                _ => None,
            }
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
