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

    fun detectAndCropFace(inputPath: String, outputPath: String, debugMode: Boolean = false): Boolean {
        try {
            // Load the input image
            val image = Imgcodecs.imread(inputPath)
            if (image.empty()) {
                println("Error: Could not load image from $inputPath")
                return false
            }

            println("Image dimensions: ${image.cols()} x ${image.rows()}")

            // Convert to grayscale for face detection
            val grayImage = Mat()
            Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY)

            // Apply histogram equalization to improve contrast (helpful for ID photos)
            val equalizedImage = Mat()
            Imgproc.equalizeHist(grayImage, equalizedImage)

            // Load the face cascade classifier
            val faceCascade = CascadeClassifier()
            val cascadePath = getHaarCascadePath()

            if (!faceCascade.load(cascadePath)) {
                println("Error: Could not load face cascade classifier")
                return false
            }

            // Detect faces with parameters optimized for ID documents
            val faces = MatOfRect()
            faceCascade.detectMultiScale(
                equalizedImage, // Use equalized image for better detection
                faces,
                1.05,    // scaleFactor (smaller = more thorough detection)
                5,       // minNeighbors (higher = more strict detection)
                0,       // flags
                Size((grayImage.cols() * 0.2).toDouble(), (grayImage.rows() * 0.2).toDouble()),  // minSize (20% of image)
                Size((grayImage.cols() * 0.8).toDouble(), (grayImage.rows() * 0.8).toDouble())   // maxSize (80% of image)
            )

            val faceArray = faces.toArray()

            if (debugMode) {
                println("Number of faces detected: ${faceArray.size}")
                faceArray.forEachIndexed { index, rect ->
                    println("Face $index: x=${rect.x}, y=${rect.y}, width=${rect.width}, height=${rect.height}")
                }
            }

            if (faceArray.isEmpty()) {
                println("No faces detected in the image")
                println("Try adjusting detection parameters or check if image contains a clear frontal face")
                return false
            }

            // Use the largest detected face (most likely to be the main subject in ID photo)
            val face = faceArray.maxByOrNull { it.width * it.height } ?: faceArray[0]
            println("Face detected at: x=${face.x}, y=${face.y}, width=${face.width}, height=${face.height}")

            // Add some padding around the face for better cropping
            val padding = (face.width * 0.2).toInt() // 20% padding
            val expandedX = maxOf(0, face.x - padding)
            val expandedY = maxOf(0, face.y - padding)
            val expandedWidth = minOf(image.cols() - expandedX, face.width + 2 * padding)
            val expandedHeight = minOf(image.rows() - expandedY, face.height + 2 * padding)

            val expandedFace = Rect(expandedX, expandedY, expandedWidth, expandedHeight)

            // Crop the face from the original color image
            val faceROI = Mat(image, expandedFace)

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
    val success = detector.detectAndCropFace(inputPath, outputPath, debugMode = true)

    if (success) {
        println("Face detection and cropping completed successfully!")
    } else {
        println("Face detection and cropping failed!")
    }
}