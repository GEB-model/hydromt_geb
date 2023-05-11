def downscale(data, factor):
    return data.repeat(factor, axis=-2).repeat(factor, axis=-1)