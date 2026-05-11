use anyhow::Result;
use clap::{Parser, Subcommand};
use splat_format::ScenePackageV3;
use std::path::PathBuf;

#[derive(Parser)]
struct Args { #[command(subcommand)] cmd: Cmd }

#[derive(Subcommand)]
enum Cmd {
    Inspect { pack: PathBuf },
    Validate { pack: PathBuf },
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Inspect { pack } => {
            let bytes = std::fs::read(pack)?;
            let scene = ScenePackageV3::decode(&bytes)?;
            println!("{}", serde_json::to_string_pretty(&scene.meta)?);
        }
        Cmd::Validate { pack } => {
            let bytes = std::fs::read(pack)?;
            let scene = ScenePackageV3::decode(&bytes)?;
            scene.validate()?;
            println!("ok: {} gaussians", scene.meta.gaussian_count);
        }
    }
    Ok(())
}
