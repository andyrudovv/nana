use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("no {0} device available")]
    NoDevice(&'static str),

    #[error("audio device: {0}")]
    Device(String),

    #[error("audio stream: {0}")]
    Stream(String),

    #[error("unsupported capture format: {0}")]
    UnsupportedFormat(String),

    #[error("wake word: {0}")]
    Wake(String),

    #[error("voice activity detection: {0}")]
    Vad(String),

    #[error("speech recognition: {0}")]
    Asr(String),

    #[error("language model: {0}")]
    Llm(String),

    #[error("model file not found: {0}")]
    ModelMissing(std::path::PathBuf),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
