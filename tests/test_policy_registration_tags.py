from mlops.tracking import normalize_policy_tag


def test_normalize_policy_tag_maps_known_contextual_policies():
    assert normalize_policy_tag("linucb") == "LinUCB"
    assert normalize_policy_tag("linear_thompson_sampling") == "LinTS"
