use bsp_monte_carlo::{BspMonteCarlo, Tolerance};

use std::collections::HashMap;

fn main() {
    #[cfg(debug_assertions)]
    eprintln!("!!! You are running on debug mode !!!");

    let mut plot = Plot::new(100.0);

    let mut mc = BspMonteCarlo::new(rand::thread_rng());
    mc.max_samples = 10_000_000;
    mc.tol = Tolerance::CI95Rel(0.0);
    let (est, ci) = mc.volume(
        |xs| {
            plot.add(xs[0], xs[1]);
            xs.into_iter().map(|x| x * x).sum::<f64>().sqrt() <= 10.0
        },
        &[-10.5; 2],
        &[10.5; 2],
    );

    plot.write(|x, y| (x * x + y * y).sqrt() <= 10.0, "./samples.png");
    println!("{:.3}(+/-){:.3}", est, ci);
    println!("> ./samples.png")
}

struct Plot {
    scale: f64,
    counter: HashMap<(i64, i64), usize>,
}

impl Plot {
    fn new(scale: f64) -> Plot {
        Plot {
            scale,
            counter: HashMap::new(),
        }
    }

    fn add(&mut self, x: f64, y: f64) {
        let ix = (x * self.scale).floor() as i64;
        let iy = (y * self.scale).floor() as i64;
        *self.counter.entry((ix, iy)).or_default() += 1;
    }

    fn write(&self, mut back: impl FnMut(f64, f64) -> bool, path: &str) {
        let min_x = self.counter.keys().map(|(x, _)| x).min().unwrap();
        let min_y = self.counter.keys().map(|(_, y)| y).min().unwrap();
        let max_x = self.counter.keys().map(|(x, _)| x).max().unwrap();
        let max_y = self.counter.keys().map(|(_, y)| y).max().unwrap();
        let max_count = *self.counter.values().max().unwrap();

        let mut img = image::RgbImage::new((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
        for (x, y, color) in img.enumerate_pixels_mut() {
            let g = if back(
                (x as i64 + min_x) as f64 / self.scale,
                (y as i64 + min_y) as f64 / self.scale,
            ) {
                64
            } else {
                0
            };
            *color = image::Rgb([255 - g, 255, 255 - g]);
        }

        for (&(x, y), &c) in self.counter.iter() {
            let r = 256.0 * c as f32 / max_count as f32;
            let r = r.clamp(0.0, 255.0).floor() as u8;
            let g = if back(x as f64 / self.scale, y as f64 / self.scale) {
                64
            } else {
                0
            };
            *img.get_pixel_mut((x - min_x) as u32, (y - min_y) as u32) =
                image::Rgb([r, g, 255 - r]);
        }

        img.save(path).unwrap();
    }
}
