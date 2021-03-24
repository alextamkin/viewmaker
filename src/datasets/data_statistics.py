def get_data_mean_and_stdev(dataset):
    if dataset == 'meta_aircraft':
        mean = [0.486, 0.507, 0.525] 
        std  = [0.266, 0.260, 0.276]
    elif dataset == 'meta_cu_birds':
        mean = [0.483, 0.491, 0.424] 
        std  = [0.228, 0.224, 0.259]
    elif dataset == 'meta_dtd':
        mean = [0.533, 0.474, 0.426]
        std  = [0.261, 0.250, 0.259]
    elif dataset == 'meta_fashionmnist':
        mean = [0.348, 0.348, 0.348] 
        std  = [0.347, 0.347, 0.347]
    elif dataset == 'meta_fungi':
        mean = [0.452, 0.421, 0.344]
        std  = [0.249, 0.237, 0.242]
    elif dataset == 'meta_mnist':
        mean = [0.170, 0.170, 0.170]
        std  = [0.320, 0.320, 0.320]
    elif dataset == 'meta_mscoco':
        mean = [0.408, 0.377, 0.352]
        std  = [0.269, 0.260, 0.261]
    elif dataset == 'meta_traffic_sign':
        mean = [0.335, 0.291, 0.295]
        std  = [0.267, 0.249, 0.251]
    elif dataset == 'meta_vgg_flower':
        mean = [0.518, 0.410, 0.329]
        std  = [0.296, 0.249, 0.285]
    elif dataset == 'mscoco':
        mean = [0.408, 0.377, 0.352]
        std  = [0.269, 0.260, 0.261]
    elif dataset == 'celeba':
        mean = [0.515, 0.417, 0.366]
        std  = [0.301, 0.272, 0.268]
    elif dataset == 'lsun':
        mean = [0.588, 0.523, 0.468]
        std  = [0.249, 0.260, 0.273]
    elif dataset == 'retinopathy':
        mean = [0.444, 0.307, 0.219]
        std  = [0.275, 0.202, 0.169]
    elif dataset == 'chexpert':
        mean = [128./256.]
        std  = [64./256.]
    elif dataset == 'chexpert_customaug':
        mean = [128./256.]
        std  = [64./256.]
    elif dataset == 'satellite':
        mean = [0.377, 0.391, 0.363]
        std  = [0.203, 0.189, 0.186]
    elif dataset == 'ham10k':
        mean = [0.764, 0.537, 0.561]
        std  = [0.137, 0.158, 0.176]
    else:
        raise Exception(f'Dataset {dataset} not supported.')

    return mean, std
