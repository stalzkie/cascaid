from cascaid.auth.passwords import hash_password, verify_password


def test_verify_password_accepts_the_correct_password():
    hashed = hash_password("correct horse battery staple")

    assert verify_password("correct horse battery staple", hashed) is True


def test_verify_password_rejects_the_wrong_password():
    hashed = hash_password("correct horse battery staple")

    assert verify_password("wrong password", hashed) is False


def test_hash_password_uses_a_random_salt_so_repeated_hashes_differ():
    first = hash_password("same password")
    second = hash_password("same password")

    assert first != second
