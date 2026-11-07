//! Incremental checking retains the whole-program rules, including constant indices.
use xetal_gpu_xir::{check, check::check_value, text};

#[test]
fn incremental_and_whole_checks_agree_and_keep_constant_lookup_refusals() {
    for source in [
        "%x = const f64 [2 3] 1.0\n%i = const i64 [2] 1 2\n%y = lookup %i %x\noutput %y\n",
        "%x = const f64 [2 3] 1.0\n%i = const i64 [2] 1 3\n%y = lookup %i %x\noutput %y\n",
    ] {
        let program = text::parse(source).unwrap();
        let mut types = Vec::new();
        let mut error = None;
        for (i, value) in program.values.iter().enumerate() {
            match check_value(value, &types, &program.values[..i]) {
                Ok(ty) => types.push(ty),
                Err(e) => {
                    error = Some(e);
                    break;
                }
            }
        }
        match error {
            None => assert_eq!(types, check(program).unwrap().types),
            Some(e) => assert_eq!(check(program).unwrap_err(), e),
        }
    }
    let program = text::parse("%x = input f64 []\n%y = map add %x %x\noutput %y\n").unwrap();
    assert!(check_value(&program.values[1], &[], &[])
        .unwrap_err()
        .0
        .contains("does not precede"));
}
