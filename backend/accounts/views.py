from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import UserRegistrationSerializer, UserLoginSerializer
from django.contrib.auth import authenticate

class BaseView(APIView):
    authentication_classes = []
    permission_classes = []

class RegistrationView(BaseView):
    def post(self, request):
        serializer = UserRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            #check is user already exists
            if serializer.is_email_exists() or serializer.is_username_exists():
                return Response({'message': 'User already exist. Please verify email.'
            }, status=status.HTTP_400_BAD_REQUEST)
            user = serializer.save()
            # Send verification email here (we'll implement later)
            return Response({
                'message': 'User registered successfully. Please verify email.'
            }, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(BaseView):

    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        
        if serializer.is_valid():
            user = authenticate(
                username=serializer.validated_data['email'],
                password=serializer.validated_data['password']
            )
            if user:
                refresh = RefreshToken.for_user(user)
                return Response({
                    'refresh': str(refresh),
                    'access': str(refresh.access_token),
                    'user_type': user.user_type,
                    'is_admin': user.is_admin,
                })
            return Response({'error': 'Invalid credentials'}, status=401)
        return Response(serializer.errors, status=400)