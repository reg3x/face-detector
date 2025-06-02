import org.opencv.core.*
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File

class FaceDetector {
    
    init {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally()
    }
    
    fun detectAndCropFace(inputPath: String, outputPath: String): Boolean {
        try {
            // Load the input image
            val image = Imgcodecs.imread(inputPath)
            if (image.empty()) {
                println("Error: Could not load image from $inputPath")
                return false
            }
            
            // Convert to grayscale for face detection
            val grayImage = Mat()
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY)
            
            // Load the face cascade classifier
            val faceCascade = CascadeClassifier()
            val cascadePath = getHaarCascadePath()
            
            if (!faceCascade.load(cascadePath)) {
                println("Error: Could not load face cascade classifier")
                return false
            }
            
            // Detect faces
            val faces = MatOfRect()
            faceCascade.detectMultiScale(
                grayImage,
                faces,
                1.1,     // scaleFactor
                3,       // minNeighbors
                0,       // flags
                Size(30.0, 30.0),  // minSize
                Size()   // maxSize (empty = no limit)
            )
            
            val faceArray = faces.toArray()
            
            if (faceArray.isEmpty()) {
                println("No faces detected in the image")
                return false
            }
            
            // Use the first detected face
            val face = faceArray[0]
            println("Face detected at: x=${face.x}, y=${face.y}, width=${face.width}, height=${face.height}")
            
            // Crop the face from the original color image
            val faceROI = Mat(image, face)
            
            // Save the cropped face
            val success = Imgcodecs.imwrite(outputPath, faceROI)
            
            if (success) {
                println("Face successfully saved to: $outputPath")
            } else {
                println("Error: Failed to save cropped face")
            }
            
            return success
            
        } catch (e: Exception) {
            println("Error during face detection: ${e.message}")
            e.printStackTrace()
            return false
        }
    }
    
    private fun getHaarCascadePath(): String {
        // You need to download haarcascade_frontalface_alt.xml from OpenCV
        // and place it in your resources or specify the full path
        return "haarcascade_frontalface_alt.xml"
    }
}

fun main(args: Array<String>) {
    if (args.size < 2) {
        println("Usage: kotlin FaceDetector <input_image_path> <output_image_path>")
        println("Example: kotlin FaceDetector input.jpg face_output.jpg")
        return
    }
    
    val inputPath = args[0]
    val outputPath = args[1]
    
    // Check if input file exists
    if (!File(inputPath).exists()) {
        println("Error: Input file does not exist: $inputPath")
        return
    }
    
    // Create face detector and process image
    val detector = FaceDetector()
    val success = detector.detectAndCropFace(inputPath, outputPath)
    
    if (success) {
        println("Face detection and cropping completed successfully!")
    } else {
        println("Face detection and cropping failed!")
    }
}