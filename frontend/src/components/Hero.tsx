
import { Button } from "@/components/ui/button";
const Hero = () => {
  const userType = localStorage.getItem('user_type');
  return (
    <div className="bg-gradient-to-r from-white-600 to-teal-700 text-black py-16">
      <div className="container mx-auto px-6 flex flex-col md:flex-row items-center">
        <div className="flex flex-col w-full md:w-1/2 justify-center items-start py-8 animate-fade-in">
          <h1 className="text-4xl md:text-5xl font-bold leading-tight mb-4">
            Detect Plagiarism & AI Content with Precision
          </h1>
          <p className="leading-normal text-xl mb-8">
            {
              userType === 'student'
                ? "View your document metrics and ensure originality with our advanced detection tools."
                : "Docuverify empowers you to analyze documents for plagiarism and AI-generated content, ensuring authenticity and integrity."


            }
          </p>
          <div className="flex flex-col sm:flex-row justify-center md:justify-start w-full">
            <Button
              className="py-6 px-8 bg-teal-700 text-white hover:bg-teal-600 rounded-lg font-bold mb-4 sm:mb-0 sm:mr-4"
              onClick={() => document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' })}
            >
              {userType !== 'student' ? 'Upload Document' : 'View Document Metrics'}
            </Button>
            <Button
              className="py-6 px-8 bg-teal-700 text-white hover:bg-teal-600 rounded-lg font-bold mb-4 sm:mb-0 sm:mr-4"
            >
              Learn More
            </Button>
          </div>
        </div>


        <div className="w-full py-6 flex justify-center">
          <div className="w-full max-w-2xl bg-white bg-opacity-10 rounded-xl backdrop-filter backdrop-blur-md border border-white border-opacity-20 shadow-2xl flex flex-col md:flex-row items-center justify-center animate-slide-up p-4">
            <div className="p-4 md:p-8 text-center flex-1">
              <div className="text-3xl md:text-5xl font-bold mb-2 md:mb-4 text-teal-600">
                99.8%
              </div>
              <p className="text-sm md:text-lg">Detection Accuracy</p>
            </div>

            <div className="hidden md:block border-r border-white border-opacity-20 h-20 mx-2 md:mx-6"></div>
            <div className="md:hidden w-full border-t border-white border-opacity-20 my-4"></div>

            <div className="p-4 md:p-8 text-center flex-1">
              <div className="text-3xl md:text-5xl font-bold mb-2 md:mb-4 text-teal-600">
                3M+
              </div>
              <p className="text-sm md:text-lg">Documents Analyzed</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
