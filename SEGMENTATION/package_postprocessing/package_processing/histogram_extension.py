import numpy as np
def histogram_extension(sample_rec, interval):
 
    sample_rec = np.array(sample_rec)
    delta = interval[1] - interval[0]
    sample_rec  = sample_rec - sample_rec.min()
    sample_rec  = sample_rec * (delta / sample_rec.max())
    sample_rec  = sample_rec + interval[0]

    return sample_rec
