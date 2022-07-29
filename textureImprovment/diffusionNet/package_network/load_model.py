from package_network.diffusionUnet    import Unet

# -----------------------------------------------------------------------------------------------------------------------
def load_model(p):
    
    param_net = {
        "input_nc":     p.INPUT_NC,
        "output_nc":    p.OUTPUT_NC,
        "n_layers":     p.NB_LAYERS,
        "ngf":          p.NGF,
        "kernel_size":  p.KERNEL_SIZE,
        "padding":      p.PADDING,
        "use_bias":     p.USE_BIAS,
        "time_emb_dim": p.TIME_EMB_DIM}
    
    model = Unet(**param_net)

    return model
