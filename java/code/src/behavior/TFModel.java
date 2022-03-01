package behavior;

import org.tensorflow.*;

public class TFModel {

    public static void main(String[] args) {
        final String modelDir = "D:\\Repos\\DungeonRL\\py\\code\\model-tf\\agent-200";
        final String modelTag = "serve";
        final String inputOp  = "serving_default_args_0";
        final String maskOp   = "serving_default_mask";
        final String detOp    = "serving_default_deterministic";
        final String outputOp = "StatefulPartitionedCall";

        SavedModelBundle b = SavedModelBundle.load(modelDir, modelTag);
        Session s = b.session();

        Tensor<Float> input           = Tensor.create(new float[][] {{2.0f, 2.0f}}, Float.class);
        Tensor<Boolean> mask          = Tensor.create(new boolean[][] {{true, true, true, true}}, Boolean.class);
        Tensor<Boolean> deterministic = Tensor.create(true, Boolean.class);

        Tensor<?> result = s.runner()
                .feed(inputOp, input)
                .feed(maskOp, mask)
                .feed(detOp, deterministic)
                .fetch(outputOp)
                .run().get(0);

        long action = result.copyTo(new long[1])[0];

        result.close();
        deterministic.close();
        mask.close();
        input.close();

        s.close();
        b.close();
    }

}
