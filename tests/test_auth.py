import logging

import pytest

import auth
from auth import AuthError, AuthManager, LockedOutError, NotAuthenticatedError

PASSWORD = "correct horse battery"
HASH = auth.hash_password(PASSWORD, rounds=4)  # low cost only to keep tests fast


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(clock=None, sleeps=None, **kw):
    clock = clock or FakeClock()
    sleeps = sleeps if sleeps is not None else []
    params = dict(password_hash=HASH, session_timeout_minutes=30, max_failed_logins=3,
                  login_delay_seconds=2, lockout_minutes=5)
    params.update(kw)
    return AuthManager(clock=clock, sleep=sleeps.append, **params), clock, sleeps


# ---- hashing
def test_hash_is_bcrypt_and_not_plaintext():
    assert HASH.startswith("$2") and PASSWORD not in HASH


def test_same_password_gives_different_hashes():
    assert auth.hash_password(PASSWORD, rounds=4) != HASH  # random salt


def test_verify_correct_and_wrong():
    assert auth.verify_password(PASSWORD, HASH)
    assert not auth.verify_password("wrong password!!", HASH)


@pytest.mark.parametrize("pw,h", [("", HASH), (PASSWORD, ""), (PASSWORD, "not-a-hash"), ("x" * 100, HASH)])
def test_verify_bad_input_returns_false(pw, h):
    assert auth.verify_password(pw, h) is False


def test_hash_rejects_overlong_password():
    with pytest.raises(ValueError):
        auth.hash_password("x" * 73)


# ---- login
def test_correct_password_returns_token():
    m, _, sleeps = make()
    token = m.login(PASSWORD)
    assert token and m.is_authenticated(token) and sleeps == []


def test_wrong_password_rejected_with_delay():
    m, _, sleeps = make()
    with pytest.raises(AuthError):
        m.login("wrong password!!")
    assert sleeps == [2]


def test_no_hash_configured_disables_login():
    m, _, _ = make(password_hash="")
    with pytest.raises(AuthError):
        m.login(PASSWORD)


def test_lockout_after_max_failures_even_for_correct_password():
    m, clock, _ = make()
    for _ in range(3):
        with pytest.raises(AuthError):
            m.login("wrong password!!")
    with pytest.raises(LockedOutError):
        m.login(PASSWORD)  # correct, but locked
    clock.t += 5 * 60 + 1
    assert m.is_authenticated(m.login(PASSWORD))  # lock expired


def test_success_resets_failure_counter():
    m, _, _ = make()
    for _ in range(2):
        with pytest.raises(AuthError):
            m.login("wrong password!!")
    m.login(PASSWORD)
    for _ in range(2):  # would be locked if the counter had not reset
        with pytest.raises(AuthError) as e:
            m.login("wrong password!!")
        assert not isinstance(e.value, LockedOutError)


# ---- sessions
def test_session_expires_after_idle_timeout():
    m, clock, _ = make()
    token = m.login(PASSWORD)
    clock.t += 31 * 60
    assert not m.is_authenticated(token)
    assert not m.is_authenticated(token)  # stays invalid


def test_activity_extends_session():
    m, clock, _ = make()
    token = m.login(PASSWORD)
    for _ in range(3):
        clock.t += 20 * 60
        assert m.is_authenticated(token)


def test_background_polling_does_not_extend_session():
    m, clock, _ = make()
    token = m.login(PASSWORD)
    for _ in range(3):  # a page polling every few minutes with refresh=False
        clock.t += 10 * 60
        assert m.is_authenticated(token, refresh=False)  # still within 30 min of the login
    clock.t += 5 * 60
    assert not m.is_authenticated(token, refresh=False)  # 35 minutes after the last REAL action


def test_forged_or_missing_token_rejected():
    m, _, _ = make()
    m.login(PASSWORD)
    assert not m.is_authenticated("forged-token")
    assert not m.is_authenticated(None)
    assert not m.is_authenticated("")


def test_logout_invalidates_token():
    m, _, _ = make()
    token = m.login(PASSWORD)
    m.logout(token)
    assert not m.is_authenticated(token)


def test_require_admin_protects_actions():
    m, _, _ = make()
    with pytest.raises(NotAuthenticatedError):
        m.require_admin(None)
    token = m.login(PASSWORD)
    m.require_admin(token)  # no exception


def test_password_and_hash_never_logged(caplog):
    m, _, _ = make()
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(AuthError):
            m.login("secret-wrong-pw-123")
        m.login(PASSWORD)
    text = caplog.text
    assert "secret-wrong-pw-123" not in text and PASSWORD not in text and HASH not in text


def test_hash_survives_dotenv_roundtrip(tmp_path):
    """The '$' characters in a bcrypt hash must not be mangled when loaded from .env."""
    from dotenv import dotenv_values
    f = tmp_path / ".env"
    f.write_text(f"ADMIN_PASSWORD_HASH='{HASH}'\n", encoding="utf-8")
    loaded = dotenv_values(f)["ADMIN_PASSWORD_HASH"]
    assert loaded == HASH and auth.verify_password(PASSWORD, loaded)
