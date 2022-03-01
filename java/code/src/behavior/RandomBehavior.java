package behavior;

import java.util.List;
import java.util.Random;

public class RandomBehavior implements IBehavior<List<Integer>, Integer> {

    private int bound;
    private Random rand;

    public RandomBehavior(int numActions) {
        bound = numActions;
        rand = new Random();
    }

    @Override
    public Integer nextAction(List<Integer> state) {
        return rand.nextInt(bound);
    }
}
