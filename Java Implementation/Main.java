import java.io.IOException;
import java.util.*;

public class Main {

    public static void main(String[] args) throws IOException {
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("/Users/bhara/Desktop/javaimp/data/train-images.idx3-ubyte", "/Users/bhara/Desktop/javaimp/data/train-labels.idx1-ubyte");
        //printMnistMatrix(mnistMatrix[mnistMatrix.length - 1]);
        mnistMatrix = new MnistDataReader().readData("/Users/bhara/Desktop/javaimp/data/train-images.idx3-ubyte", "/Users/bhara/Desktop/javaimp/data/train-labels.idx1-ubyte");
        //printMnistMatrix(mnistMatrix[59999]);


        int[] trainLabels;
        int[][] trainMatrices;
        trainLabels = new int[59999];
        trainMatrices = new int[59999][784];
        int[] testLabels;
        int[][] testMatrices;
        testLabels = new int[9999];
        trainMatrices = new int[9999][784];

        for(int i = 0; i < 59999; i++){
            trainLabels[i] = mnistMatrix[i].getLabel();
        }
        mnistMatrix = new MnistDataReader().readData("/Users/bhara/Desktop/javaimp/data/t10k-images.idx3-ubyte", "/Users/bhara/Desktop/javaimp/data/t10k-labels.idx1-ubyte");
        for(int i = 0; i < 9999; i++){
            testLabels[i] = mnistMatrix[i].getLabel();
            for j in length(data[i][j])
                for k in lenght(data[i][j[k]])
                    new array 
        }
    }

    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }

    private static void insertMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}