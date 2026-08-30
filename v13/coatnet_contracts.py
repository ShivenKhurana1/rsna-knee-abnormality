"""Frozen CoAtNet input contracts verified against the author's public releases.

The cache resolution is not the network resolution. Quantizing at 336 then
upsampling to 384 cannot reproduce a corpus quantized directly at 384.
"""

import hashlib as _v13_contract_hashlib
import json as _v13_contract_json


V13_COATNET_CONTRACTS = {
    'raptor_ft_coatnet_v5_full_swa.pt': {
        'cache_img': 336, 'crop_mm': 140.0, 'span_lo': 0.02, 'span_hi': 0.98,
        'k_eval': 62, 'model_res': 384,
        'source': 'https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan',
    },
    'raptor_ft_coatnet_v10_full.pt': {
        'cache_img': 384, 'crop_mm': 140.0, 'span_lo': 0.02, 'span_hi': 0.98,
        'k_eval': 62, 'model_res': 384,
        'source': 'https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense',
    },
    'raptor_ft_coatnet_v4_full.pt': {
        'cache_img': 336, 'crop_mm': 140.0, 'span_lo': 0.06, 'span_hi': 0.94,
        'k_eval': 42, 'model_res': 384,
        'source': 'https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model',
    },
}


def v13_coatnet_contract(filename):
    if filename not in V13_COATNET_CONTRACTS:
        raise ValueError(f'Unknown CoAtNet preprocessing contract: {filename}')
    return dict(V13_COATNET_CONTRACTS[filename])


def v13_volume_key(study_id, contract):
    # Include every contract field; over-separating is safe, sharing incompatible
    # volumes is not. Tuple encoding prevents separator/UID collisions.
    encoded = _v13_contract_json.dumps([str(study_id), contract], sort_keys=True)
    return _v13_contract_hashlib.sha256(encoded.encode()).hexdigest()


def v13_check_coatnet_checkpoint(checkpoint, expected_labels, contract):
    if list(checkpoint.get('lab', [])) != list(expected_labels):
        raise ValueError('CoAtNet checkpoint target order differs from the submission contract')
    if int(checkpoint.get('res', -1)) != contract['model_res']:
        raise ValueError('CoAtNet checkpoint resolution differs from its preprocessing contract')
