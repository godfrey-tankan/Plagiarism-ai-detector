import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LoginCredentials } from "@/types/auth";
import { useToast } from "@/components/ui/use-toast";
import { loginUser } from "@/lib/api"
import Loader from "@/components/Loader.jsx"

const LoginForm = () => {
  const [credentials, setCredentials] = useState<LoginCredentials>({
    email: "",
    password: "",
  });
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false)

  const navigate = useNavigate();
  const { toast } = useToast();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await loginUser(credentials);

      if (response.data.access && response.data.refresh) {
        localStorage.setItem('access_token', response.data.access);
        localStorage.setItem('refresh_token', response.data.refresh);
        localStorage.setItem('user_type', response.data.user_type);

        toast({
          title: "Login Successful",
          description: "Welcome back!",
        });
        // if (response.data.user_type === 'supervisor') {
        //   navigate('/dashboard');
        //   return; // Important to return here to prevent further execution
        // }
        // if (response.data.user_type === 'student') {
        //   navigate('/student-dashboard');
        //   return; // Important to return here to prevent further execution
        // }
        // if (response.data.user_type === 'admin') {
        //   navigate('/admin-dashboard');
        //   return; // Important to return here to prevent further execution
        // }
        navigate('/dashboard');
        return; // Important to return here to prevent further execution
      }

      throw new Error('Invalid response from server');

    } catch (err) {
      let errorMessage = 'Login failed. Please check your credentials.';

      if (err.response) {
        // Handle Django backend errors
        errorMessage = err.response.data.detail || errorMessage;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
      // toast({
      //   title: "Login Failed",
      //   description: errorMessage,
      //   variant: "destructive"
      // });

    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-center">Login</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}
          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="email">
              Email
            </label>
            <Input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={credentials.email}
              onChange={(e) =>
                setCredentials({ ...credentials, email: e.target.value })
              }
              required
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium" htmlFor="password">
              Password
            </label>
            <Input
              id="password"
              type="password"
              placeholder="Enter your password"
              value={credentials.password}
              onChange={(e) =>
                setCredentials({ ...credentials, password: e.target.value })
              }
              required
            />
          </div>
          <Button
            type="submit"
            className="w-full bg-teal-700 hover:bg-teal-800 uppercase"
            disabled={isLoading}
          >
            {isLoading ? <Loader size="small" /> : 'Sign In'}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default LoginForm;