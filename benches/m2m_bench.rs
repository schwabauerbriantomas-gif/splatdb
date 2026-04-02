use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use ndarray::{Array1, Array2};
use rand::Rng;

fn random_array2(nrows: usize, ncols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..nrows * ncols).map(|_| rng.gen::<f32>()).collect();
    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

fn random_array1(n: usize) -> Array1<f32> {
    let mut rng = rand::thread_rng();
    Array1::from_vec((0..n).map(|_| rng.gen::<f32>()).collect())
}

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");
    for &n in &[100, 1000, 10000] {
        let data = random_array2(n, 64);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut km = m2m_vector_search::clustering::KMeans::new(10, 50, 42);
                km.fit(black_box(&data));
            });
        });
    }
    group.finish();
}

fn bench_encoding(c: &mut Criterion) {
    let builder = m2m_vector_search::encoding::FullEmbeddingBuilder::new();
    let pos = [1.0_f32, 2.0, 3.0];
    let col = [0.5_f32, 0.5, 0.5];
    let scale = [0.1_f32, 0.1, 0.1];
    let rot = [1.0_f32, 0.0, 0.0, 0.0];

    c.bench_function("full_embedding_single", |b| {
        b.iter(|| {
            builder.build_single(
                black_box(&pos), black_box(&col),
                black_box(0.9), black_box(&scale), black_box(&rot),
            );
        });
    });

    let n = 1000;
    let positions = random_array2(n, 3);
    let colors = random_array2(n, 3);
    let opacities = random_array1(n);
    let scales = random_array2(n, 3);
    let rotations = random_array2(n, 4);

    c.bench_function("full_embedding_batch_1k", |b| {
        b.iter(|| {
            builder.build(
                black_box(&positions), black_box(&colors),
                black_box(&opacities), black_box(&scales), black_box(&rotations),
            );
        });
    });
}

fn bench_energy(c: &mut Criterion) {
    let weights = m2m_vector_search::energy::EnergyWeights::default();
    let ef = m2m_vector_search::energy::EnergyFunction::new(weights);

    let batch = random_array2(100, 640);

    c.bench_function("e_geom_100_vectors", |b| {
        b.iter(|| {
            ef.e_geom(black_box(&batch));
        });
    });

    let mu = random_array2(1000, 640);
    let alpha: Vec<f32> = (0..1000).map(|_| rand::random::<f32>()).collect();
    let kappa: Vec<f32> = (0..1000).map(|_| rand::random::<f32>()).collect();
    let x = random_array2(100, 640);

    c.bench_function("e_splats_vectorized_100x1k", |b| {
        b.iter(|| {
            ef.e_splats_vectorized(
                black_box(&x), black_box(&mu),
                black_box(&alpha), black_box(&kappa), 1000,
            );
        });
    });
}

fn bench_geometry(c: &mut Criterion) {
    let data = random_array2(10000, 640);
    c.bench_function("normalize_sphere_10k", |b| {
        b.iter(|| {
            m2m_vector_search::geometry::normalize_sphere(black_box(&data));
        });
    });
}

fn bench_clustering_transform(c: &mut Criterion) {
    let mut km = m2m_vector_search::clustering::KMeans::new(10, 50, 42);
    let train = random_array2(1000, 64);
    km.fit(&train);

    let query = random_array2(100, 64);
    c.bench_function("kmeans_transform_100_queries", |b| {
        b.iter(|| {
            km.transform(black_box(&query));
        });
    });
}

criterion_group!(benches, bench_kmeans, bench_encoding, bench_energy, bench_geometry, bench_clustering_transform);
criterion_main!(benches);
