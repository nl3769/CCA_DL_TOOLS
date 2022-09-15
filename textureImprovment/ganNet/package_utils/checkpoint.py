import os

# -----------------------------------------------------------------------------------------------------------------------
def save_loss(set, epoch, loss, folder, substr = None):
    
    if epoch < 10:
        epoch = "0000" + str(epoch)
    elif epoch < 100:
        epoch = "000" + str(epoch)
    elif epoch < 1000:
        epoch = "00" + str(epoch)
    elif epoch < 10000:
        epoch = "0" + str(epoch)
    else:
        epoch = str(epoch)
    
    if substr is None:
        pname = set + "_" + epoch + ".txt"
    else:
        pname = set + "_" + epoch + "_" + substr + ".txt"
    
    with open(os.path.join(folder, pname), 'w') as f:
        f.write(str(loss))
