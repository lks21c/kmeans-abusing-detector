import org.junit.Test;
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
public class KmeansTest {
    @Test
    public void testKmeans() throws Exception {
        Map<Long, Double> datapoints = new HashMap<>();
        datapoints.put(1495738800000L, 354.0);
        datapoints.put(1495739400000L, 708.0);
        datapoints.put(1495740000000L, 0.0);
        datapoints.put(1495740600000L, 354.0);
        datapoints.put(1495741200000L, 354.0);
        datapoints.put(1495741800000L, 354.0);
        datapoints.put(1495742400000L, 1854.0);
        datapoints.put(1495743000000L, 2354.0);

        try {
            KmeansAbusingDetector.trainModel(datapoints, 1);
        } catch (Exception e) {
            throw new UnsupportedOperationException("Cluster creation unsuccessful");
        }

        Map<Long, Double> predictions = KmeansAbusingDetector.predictAnomalies(datapoints);
        Map<Long, Double> predictionsNormalized = KmeansAbusingDetector.normalizePredictions(predictions);

        System.out.println("predictions = " + predictions.toString());
        System.out.println("predictionsNormalized = " + predictionsNormalized.toString());
    }
}
