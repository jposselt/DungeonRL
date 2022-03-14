package desktop;

import actor.RandomNPC;
import actor.TestNPC;
import behavior.IBehavior;
import behavior.RandomBehavior;
import behavior.TFModel;
import controller.MainController;
import level.LevelAPI;
import level.generator.dummy.DummyGenerator;
import level.generator.dungeong.graphg.NoSolutionException;
import tools.Point;

public class Game extends MainController {
    private TestNPC npc;

    @Override
    public void onLevelLoad() {
        npc.setLevel(levelAPI.getCurrentLevel());
    }

    @Override
    protected void beginFrame() {
        // TODO Auto-generated method stub
        
    }

    @Override
    protected void endFrame() {
        // TODO Auto-generated method stub
        
    }

    @Override
    protected void setup() {
        generator = new DummyGenerator();
        levelAPI = new LevelAPI(batch, painter, generator, this);

        final String texture = "character/monster/chort_idle_anim_f0.png";
        final String modelDir = "D:\\Repos\\DungeonRL\\models\\tf\\ppo-200";
        IBehavior<Point, Integer> behavior = new TFModel(modelDir);
        npc = new TestNPC(batch, painter, texture, behavior);

        entityController.add(npc);
        camera.follow(npc);

        try {
            levelAPI.loadLevel();
        } catch (NoSolutionException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        Launcher.run(new Game());
    }
}