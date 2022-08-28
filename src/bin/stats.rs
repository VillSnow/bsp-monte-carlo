use bsp_monte_carlo::BspMonteCarlo;

use rayon::prelude::*;
use std::f64::consts::PI;

fn main() {
    const N: usize = 1000;

    for dim in 1..=10usize {
        #[cfg(debug_assertions)]
        eprintln!("!!! You are running on debug mode !!!");

        let r = 1.0;
        println!("[{}-sphere]", dim);
        let true_value = nsphere_actual(r, dim as i32);

        let ys = (0..N)
            .into_par_iter()
            .map(|_| {
                let mut mc = BspMonteCarlo::new(rand::thread_rng());
                mc.step_samples = 10_000;
                mc.volume(
                    |xs| xs.into_iter().map(|x| x * x).sum::<f64>().sqrt() <= r,
                    &vec![-1.1 * r; dim],
                    &vec![1.1 * r; dim],
                )
            })
            .collect::<Vec<_>>();

        let rms = get_mean(ys.iter().map(|&(y, _)| (y - true_value).powi(2))).sqrt();

        let mean_ci = get_mean(ys.iter().map(|&(_, ci)| ci));

        let acc = get_mean(ys.iter().map(|&(y, ci)| {
            if (y - true_value).abs() <= ci {
                1.0
            } else {
                0.0
            }
        }));

        println!("true = {:9.3}", true_value);
        println!("rms  = {:9.3}", rms);
        println!("ci   = {:9.3}", mean_ci);
        println!("acc  = {:9.3}%", acc * 100.0);
    }
}

fn get_mean(xs: impl IntoIterator<Item = f64>) -> f64 {
    let mut x1 = 0.0;
    let mut x0 = 0.0;
    for x in xs {
        x1 += x;
        x0 += 1.0
    }
    x1 / x0
}

fn gamma(z: f64) -> f64 {
    if z < 0.0 {
        unimplemented!();
    }
    if z == 1.0 {
        return 1.0;
    }
    if z == 0.5 {
        return PI.sqrt();
    }

    (z - 1.0) * gamma(z - 1.0)
}

fn nsphere_actual(r: f64, dim: i32) -> f64 {
    PI.powf(dim as f64 / 2.0) * r.powi(dim) / gamma(dim as f64 / 2.0 + 1.0)
}
