config = {
    "dataset_name": "hyunlord/query_image_anchor_positive_large_384",
    "cache_dir": "./.cache",
    "model_name": "google/siglip2-base-patch16-384",

    "hash_hidden_dim": 512,
    "margin": 0.242047,
    "lambda_ortho": 0.197038,
    "lambda_lcs": 1.137855,
    "lambda_cons": 0.1,
    "lambda_quant": 0.01,

    "batch_groups": 4,
    "images_per_group": 10,
    "image_size": 384,
    "learning_rate": 0.000020,
    "epochs": 50,
    "num_workers": 28,
    "seed": 42,

    "bit_list": [8, 16, 32, 48, 64, 128]
}