use env_logger::Builder;
use log::LevelFilter;
use std::io::Write;

pub fn build_pace_logger_for_level(level: LevelFilter) {
    let mut builder = Builder::from_default_env();
    builder
        .format(|buf, record| writeln!(buf, "c {} - {}", record.level(), record.args()))
        .filter(None, level)
        .init();
}

pub fn build_pace_logger() {
    let mut builder = Builder::from_default_env();
    builder
        .format(|buf, record| writeln!(buf, "c {} - {}", record.level(), record.args()))
        .init();
}
