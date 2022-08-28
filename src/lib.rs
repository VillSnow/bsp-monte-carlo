use rand::prelude::*;

pub enum Tolerance {
    CI95Abs(f64),
    CI95Rel(f64),
}

pub struct BspMonteCarlo<Rng: rand::Rng> {
    pub tol: Tolerance,
    pub step_samples: usize,
    pub max_samples: usize,
    pub rng: Rng,
}

impl<Rng: rand::Rng> BspMonteCarlo<Rng> {
    pub fn new(rng: Rng) -> Self {
        BspMonteCarlo {
            tol: Tolerance::CI95Rel(0.01),
            step_samples: 1000,
            max_samples: 1_000_000,
            rng,
        }
    }

    pub fn volume(
        &mut self,
        mut f: impl FnMut(&[f64]) -> bool,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
    ) -> (f64, f64) {
        assert_eq!(lower_bounds.len(), upper_bounds.len());
        let dim = lower_bounds.len();

        let mut n = self.max_samples;

        let mut root = Node::new(
            &mut f,
            lower_bounds,
            upper_bounds,
            n.min(self.step_samples),
            &mut self.rng,
        );
        n -= n.min(self.step_samples);

        for i in 0.. {
            if n == 0 {
                break;
            }

            let tol_meets = match self.tol {
                Tolerance::CI95Abs(tol) => root.ci95() <= tol,
                Tolerance::CI95Rel(tol) => root.ci95() <= tol * root.iest,
            };

            if tol_meets {
                break;
            } else {
                root.split(&mut f, n.min(self.step_samples), i % dim, &mut self.rng);
                n -= n.min(self.step_samples);
            }
        }

        (root.iest, root.ci95())
    }
}

#[derive(Clone)]
struct Node {
    est: f64,
    iest: f64,
    var: f64,
    ivar: f64,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    branches: Option<Box<(Self, Self)>>,
}

impl Node {
    fn new(
        f: &mut impl FnMut(&[f64]) -> bool,
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        samples: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let dim = lower_bounds.len();
        let distrs = (0..dim)
            .map(|i| rand::distributions::Uniform::new(lower_bounds[i], upper_bounds[i]))
            .collect::<Vec<_>>();

        let hits = (0..samples)
            .filter(|_| {
                let xs = distrs.iter().map(|d| d.sample(rng)).collect::<Vec<_>>();
                f(&xs)
            })
            .count();

        let p = hits as f64 / samples as f64;

        let mut rect_volume = 1.0;
        for i in 0..dim {
            rect_volume *= upper_bounds[i] - lower_bounds[i];
        }

        let est = p * rect_volume;
        let var = p * (1.0 - p) * rect_volume.powi(2) / samples as f64;
        Node {
            est,
            iest: est,
            var,
            ivar: var,
            lower_bounds: lower_bounds.to_vec(),
            upper_bounds: upper_bounds.to_vec(),
            branches: None,
        }
    }

    fn ci95(&self) -> f64 {
        self.ivar.sqrt() * 1.96
    }

    fn split(
        &mut self,
        f: &mut impl FnMut(&[f64]) -> bool,
        samples: usize,
        axis: usize,
        rng: &mut impl rand::Rng,
    ) {
        if let Some(branches) = &mut self.branches {
            let (a, b) = branches.as_mut();

            if a.ivar < b.ivar {
                b.split(f, samples, axis, rng);
            } else {
                a.split(f, samples, axis, rng);
            }
        } else {
            let mut ub_a = self.upper_bounds.clone();
            let mut lb_b = self.lower_bounds.clone();

            ub_a[axis] = (self.lower_bounds[axis] + self.upper_bounds[axis]) / 2.0;
            lb_b[axis] = (self.lower_bounds[axis] + self.upper_bounds[axis]) / 2.0;

            let a = Node::new(f, &self.lower_bounds, &ub_a, samples - samples / 2, rng);
            let b = Node::new(f, &lb_b, &self.upper_bounds, samples / 2, rng);

            self.branches = Some(Box::new((a, b)));
        }

        self.update_integrated();
    }

    fn update_integrated(&mut self) {
        if let Some(branches) = &self.branches {
            let prev = self.ivar;

            let (a, b) = branches.as_ref();

            let cest = a.iest + b.iest;
            let cvar = a.ivar + b.ivar;

            let alpha = self.var / (self.var + cvar);

            self.iest = (1.0 - alpha) * self.est + alpha * cest;
            self.ivar = (1.0 - alpha).powi(2) * self.var + alpha.powi(2) * cvar;

            assert!(self.ivar <= prev, "self.ivar={}, prev={}", self.ivar, prev);
        }
    }
}
