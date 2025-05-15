
import { Button } from "@/components/ui/button";
import { FileText, Settings, User } from "lucide-react";
import { logoutUser } from "@/lib/api";
import { Link } from "react-router-dom";


const Navbar = () => {
  const isAuthenticated = localStorage.getItem("access_token") !== null;
  return (
    <nav className="bg-white border-b py-4 px-6 shadow-sm sticky top-0 z-50">
      <div className="container mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <FileText className="h-6 w-6 text-teal-600" />
          <span className="text-xl font-bold text-gray-800 hover:cursor-pointer">
            <a href="/">DocuVerify</a></span>
        </div>
        {isAuthenticated && (

          <div className="hidden md:flex items-center space-x-6">
            <a href="/dashboard" className="text-gray-600 hover:text-teal-600 transition-colors">Home</a>
            <a href="/history" className="text-gray-600 hover:text-teal-600 transition-colors">History</a>
            <a href="#" className="text-gray-600 hover:text-teal-600 transition-colors">About</a>
          </div>
        )}

        <div className="flex items-center space-x-3">
          <Button variant="ghost" size="icon" className="text-gray-600 hover:text-teal-600 hover:bg-teal-50">
            <Settings className="h-5 w-5" />
          </Button>
          <Button variant="ghost" size="icon" className="text-gray-600 hover:text-teal-600 hover:bg-teal-50">
            <User className="h-5 w-5" />
          </Button>
          {isAuthenticated && (
            <Button className="bg-teal-600 text-white hover:bg-teal-700"
              onClick={logoutUser}
            >
              Logout
            </Button>
          )}
          {!isAuthenticated && (
            <>

              <Button variant="outline" className="border-teal-600 text-teal-600 hover:bg-teal-50"
              >

                <Link to="/login" className="text-teal-600 hover:text-teal-700 font-medium">
                  Login
                </Link>
              </Button>

              {/* <Button variant="outline" className="border-teal-600 text-teal-600 hover:bg-teal-50"
              > */}
              {/* <Link to="/signup" className="text-teal-600 hover:text-teal-700 font-medium">
                  Signup
                </Link> */}
              {/* </Button> */}
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
