import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Created by lks21c on 17. 5. 31.
 */
public class KmeansAbusingDetector {
    private static Instances trainingData;
    private static SimpleKMeans model;
    private static Instances clusterCentroids;
    private static int[] centroidAssignments;
    private static Map<Instance, Double> meanDistancesToCentroids;


    public static void trainModel(Map<Long, Double> metricData, int k) throws Exception {
        //Model has a single metric_value attribute
        Attribute value = new Attribute("metric_value");
        FastVector attributes = new FastVector();
        attributes.addElement(value);

        trainingData = new Instances("metric_value_data", attributes, 0);
        for (Double val : metricData.values()) {
            double[] valArray = new double[]{val};
            Instance instance = new Instance(1.0, valArray);
            trainingData.add(instance);
        }

        //Create and train the model
        model = new SimpleKMeans();
        model.setNumClusters(k);
        model.setMaxIterations(20);
        model.setPreserveInstancesOrder(true);
        model.buildClusterer(trainingData);

        clusterCentroids = model.getClusterCentroids();
        centroidAssignments = model.getAssignments();
        setMeanDistancesToCentroids();
    }

    public static Map<Long, Double> predictAnomalies(Map<Long, Double> metricData) {
        List<Double> metricDataValues = metricData.values().stream().collect(Collectors.toList());
        Map<Long, Double> predictionDatapoints = new HashMap<>();

        for (Map.Entry<Long, Double> entry : metricData.entrySet()) {
            Long timestamp = entry.getKey();
            double value = entry.getValue();
            try {
                double anomalyScore = calculateAnomalyScore(value, metricDataValues);
                predictionDatapoints.put(timestamp, anomalyScore);
            } catch (ArithmeticException e) {
                continue;
            }
        }
        return predictionDatapoints;
    }

    public static Map<Long, Double> normalizePredictions(Map<Long, Double> predictions) {
        Map<String, Double> minMax = getMinMax(predictions);
        double min = minMax.get("min");
        double max = minMax.get("max");

        Map<Long, Double> metricDataNormalized = new HashMap<>();

        if (max - min == 0.0) {
            /**
             * If (max - min) == 0.0, all data points in the predictions metric
             * have the same value. So, all data points in the normalized metric
             * will have value 0. This avoids divide by zero operations later on.
             */
            for (Long timestamp : predictions.keySet()) {
                metricDataNormalized.put(timestamp, 0.0);
            }
        } else {
            double normalizationConstant = 100.0 / (max - min);

            for (Map.Entry<Long, Double> entry : predictions.entrySet()) {
                Long timestamp = entry.getKey();
                Double value = entry.getValue();

                // Formula: normalizedValue = (rawValue - min) * (100 / (max - min))
                Double valueNormalized = (value - min) * normalizationConstant;
                metricDataNormalized.put(timestamp, valueNormalized);
            }
        }
        return metricDataNormalized;
    }

    private static Map<String, Double> getMinMax(Map<Long, Double> metricData) {
        double min = 0.0;
        double max = 0.0;
        boolean isMinMaxSet = false;
        for (Double value : metricData.values()) {
            double valueDouble = value;
            if (!isMinMaxSet) {
                min = valueDouble;
                max = valueDouble;
                isMinMaxSet = true;
            } else {
                if (valueDouble < min) {
                    min = valueDouble;
                } else if (valueDouble > max) {
                    max = valueDouble;
                }
            }
        }

        Map<String, Double> minMax = new HashMap<>();
        minMax.put("min", min);
        minMax.put("max", max);
        return minMax;
    }

    private static void setMeanDistancesToCentroids() {
        meanDistancesToCentroids = new HashMap<>();
        for (int i = 0; i < clusterCentroids.numInstances(); i++) {    //For each centroid
            int countAssignedInstances = 0;
            double sumDistancesToCentroid = 0.0;
            Instance centroidInstance = clusterCentroids.instance(i);
            for (int j = 0; j < trainingData.numInstances(); j++) {       //For each data point
                if (i == centroidAssignments[j]) {
                    Instance valueInstance = trainingData.instance(j);
                    double distanceToCentroid = Math.abs(valueInstance.value(0) -
                            centroidInstance.value(0));
                    sumDistancesToCentroid += distanceToCentroid;
                    countAssignedInstances++;
                }
            }
            double meanDistanceToCentroid = sumDistancesToCentroid / countAssignedInstances;
            meanDistancesToCentroids.put(centroidInstance, meanDistanceToCentroid);
        }
    }

    public static double calculateAnomalyScore(double value, List<Double> metricDataValues) {
        int instanceIndex = metricDataValues.indexOf(value);
        Instance valueInstance = trainingData.instance(instanceIndex);
        //Centroid that is assigned to valueInstance
        Instance centroidInstance = clusterCentroids.instance(centroidAssignments[instanceIndex]);

        if (meanDistancesToCentroids.get(centroidInstance) == 0.0) {
            throw new ArithmeticException("Cannot divide by 0");
        }

        double distanceToCentroid = Math.abs(valueInstance.value(0) - centroidInstance.value(0));
        double relativeDistanceToCentroid = distanceToCentroid / meanDistancesToCentroids.get(centroidInstance);
        return relativeDistanceToCentroid;
    }
}
