package behavior;

import org.tensorflow.*;
import tools.Point;

public class TFModel implements IBehavior<Point, Integer> {
    private final Session session;

    public TFModel(String modelDir) {
        final String modelTag = "serve";
        this.session = SavedModelBundle.load(modelDir, modelTag).session();
    }

    @Override
    public Integer nextAction(Point state, boolean[] actionMask) throws IllegalArgumentException {
        final String inputOp  = "serving_default_args_0";
        final String maskOp   = "serving_default_mask";
        final String detOp    = "serving_default_deterministic";
        final String outputOp = "StatefulPartitionedCall";

        try (Tensor<Float> input           = Tensor.create(new float[][] {{state.x, state.y}}, Float.class);
             Tensor<Boolean> mask          = Tensor.create(new boolean[][] {actionMask}, Boolean.class);
             Tensor<Boolean> deterministic = Tensor.create(true, Boolean.class);
             ) {
            try (Tensor<?> result = this.session.runner()
                    .feed(inputOp, input)
                    .feed(maskOp, mask)
                    .feed(detOp, deterministic)
                    .fetch(outputOp)
                    .run().get(0);) {
                return (Integer) (int) result.copyTo(new long[1])[0];
            }
        }
    }

}
