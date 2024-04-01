#include "utils.h"

Timer::Timer(double& accumulator, bool isEnabled)
    : accumulator(accumulator), isEnabled(isEnabled) {
    if (isEnabled) {
        start = std::chrono::high_resolution_clock::now();
    }
}

// Stop the timer and update the accumulator
void Timer::Stop() {
    if (isEnabled) {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        accumulator += duration;
    }
}

void hexToString(char str[], int length)
{
  //hexadecimal characters
  char hex_characters[]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
  
  int i;
  for(i=0;i<length;i++)
  {
    str[i]=hex_characters[rand()%16];
  }
  str[length]=0;
}

cv::Scalar colorConverter(int hexValue)
{
    struct RGB rgbColor;
    rgbColor.r = ((hexValue >> 16) & 0xFF);  // Extract the RR byte
    rgbColor.g = ((hexValue >> 8) & 0xFF);   // Extract the GG byte
    rgbColor.b = ((hexValue) & 0xFF);        // Extract the BB byte
    return cv::Scalar( (int) rgbColor.r, (int) rgbColor.g, (int) rgbColor.b); 
}

std::vector<cv::Scalar> utils::colorVectorScalar(int num)
{
    int length = 6;
    char hex[length];
    std::vector<cv::Scalar> classColors;
    
    for(int i=0; i<num; i++)
    {
        hexToString(hex, length);
        unsigned int hex_value = std::stoul(hex, nullptr, 16);
        classColors.push_back( colorConverter( hex_value ) );
    }

    return classColors;
}

std::string utils::splitExtension(std::string fileNamePath){
    std::string onlyFileNamePath = fileNamePath;
    const size_t period_idx = fileNamePath.rfind('.');
    if (std::string::npos != period_idx)
    {
        onlyFileNamePath.erase(period_idx);
    }

    return onlyFileNamePath;
}

std::vector<float> utils::arrayToVector(const float *data, std::vector<int64_t> outputShape) {

	int64_t output_tensor_size = 1;
	for (auto& it : outputShape)
	{
		output_tensor_size *= it;
	}
	std::vector<float> results(output_tensor_size);
	for (unsigned i = 0; i < output_tensor_size; i++)
	{
		results[i] = data[i];
	}
    return results;
}

size_t utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

std::wstring utils::charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}

std::vector<std::string> utils::loadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);

    if (infile.good())
    {
        std::string line;
        while (getline (infile, line))
        {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}

void utils::cornerRect(cv::Mat& image, cv::Rect bbox, cv::Scalar color, int t=5, int rt=1)
{
    int xmin = bbox.x;
    int ymin = bbox.y;
    int xmax = bbox.x + bbox.width;
    int ymax = bbox.y + bbox.height;
    int l = std::max(1, int(std::min( (ymax-ymin), (xmax-xmin))*0.2));

    if (rt != 0)
        cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, rt);

    // Top Left  xmin, ymin
    cv::line(image,  cv::Point(xmin, ymin), cv::Point(xmin + l, ymin), color, t);
    cv::line(image,  cv::Point(xmin, ymin), cv::Point(xmin, ymin + l), color, t);
    // Top Right  xmax, ymin
    cv::line(image, cv::Point(xmax, ymin), cv::Point(xmax - l, ymin), color, t);
    cv::line(image, cv::Point(xmax, ymin), cv::Point(xmax, ymin + l), color, t);
    // Bottom Left  xmin, ymax
    cv::line(image, cv::Point(xmin, ymax), cv::Point(xmin + l, ymax), color, t);
    cv::line(image, cv::Point(xmin, ymax), cv::Point(xmin, ymax - l), color, t);
    // Bottom Right  xmax, ymax
    cv::line(image, cv::Point(xmax, ymax), cv::Point(xmax - l, ymax), color, t);
    cv::line(image, cv::Point(xmax, ymax), cv::Point(xmax, ymax - l), color, t);
}

void utils::drawDetectOnFrame(cv::Mat& image, std::vector<Detection>& detections,
                               const std::vector<std::string>& classNames, std::vector<cv::Scalar> classColors)
{
    for (const Detection& detection : detections)
    {
        int x = detection.box.x;
        int y = detection.box.y;

        int conf = (int)std::round(detection.conf * 100);
        int classId = detection.classId;
        std::string label = classNames[classId] + " 0." + std::to_string(conf);

        utils::cornerRect(image, detection.box, classColors[classId]);
        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.45, 2, &baseline);
        cv::rectangle(image,
                      cv::Point(x-3, y - 20), cv::Point(x + size.width, y-5),
                      classColors[classId], -1);

        cv::putText(image, label,
                    cv::Point(x, y - 8), cv::FONT_ITALIC,
                    0.45, cv::Scalar(255, 255, 255), 2);
    }
}

void utils::letterBox(const cv::Mat& image, cv::Mat& outImage,
                      const cv::Size& newShape = cv::Size(640, 640),
                      const cv::Scalar& color = cv::Scalar(114, 114, 114),
                      bool auto_ = true,
                      bool scaleFill = false,
                      bool scaleUp = true,
                      int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2] {r, r};
    int newUnpad[2] {(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void utils::scaleCoords(cv::Rect& coords, const cv::Size& imageShape, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                          (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int) (( (float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int) (( (float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int) std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int) std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int) std::round(((float)coords.width / gain));
    coords.height = (int) std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    // coords.x = utils::clip(coords.x, 0, imageOriginalShape.width);
    // coords.y = utils::clip(coords.y, 0, imageOriginalShape.height);
    // coords.width = utils::clip(coords.width, 0, imageOriginalShape.width);
    // coords.height = utils::clip(coords.height, 0, imageOriginalShape.height);
}

template <typename T>
T utils::clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

std::string utils::getFileExtension(const std::string& path)
{
    std::string ext = path.substr(path.find_last_of(".") + 1);
    return ext;
}

bool utils::isImage(const std::string& path)
{
    static const std::string extensions[] = { "jpg", "jpeg", "png", "bmp", "gif" };
    std::string extension = path.substr(path.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    return std::find(std::begin(extensions), std::end(extensions), extension) != std::end(extensions);
}