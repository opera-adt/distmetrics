transformer_latest_config = {
    'type': 'transformer',
    'patch_size': 8,
    'num_patches': 4,
    'data_dim': 128,
    'd_model': 256,
    'nhead': 4,
    'num_encoder_layers': 4,
    'dim_feedforward': 768,
    'max_seq_len': 10,
    'dropout': 0.2,
    'activation': 'relu',
}

transformer_config = {
    'type': 'transformer (space and time pos encoding)',
    'patch_size': 8,
    'num_patches': 4,
    'data_dim': 128,  # 2 * patch_size * patch_size
    'd_model': 256,
    'nhead': 4,
    'num_encoder_layers': 2,
    'dim_feedforward': 512,
    'max_seq_len': 10,
    'dropout': 0.2,
    'activation': 'relu',
}
