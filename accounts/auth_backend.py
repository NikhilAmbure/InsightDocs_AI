from django.contrib.auth.backends import ModelBackend
from django.contrib.auth import get_user_model


class EmailBackend(ModelBackend):
    """
    Authenticate users using their email address instead of username.
    Works together with the default ModelBackend.
    """

    def authenticate(self, request, username=None, email=None, password=None, **kwargs):
        UserModel = get_user_model()

        # Allow both `username` and `email` parameters, but treat them as email
        if email is None:
            email = username

        if email is None or password is None:
            return None

        try:
            user = UserModel.objects.get(email=email)
        except UserModel.DoesNotExist:
            return None

        if user.check_password(password):
            return user

        return None

    def get_user(self, user_id):
        UserModel = get_user_model()
        try:
            return UserModel.objects.get(pk=user_id)
        except UserModel.DoesNotExist:
            return None


