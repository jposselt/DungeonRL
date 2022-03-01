package desktop;

import actor.RandomNPC;
import behavior.RandomBehavior;
import controller.MainController;
import level.LevelAPI;
import level.generator.dummy.DummyGenerator;
import level.generator.dungeong.graphg.NoSolutionException;

public class Game extends MainController {
    private RandomNPC npc;

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
        npc = new RandomNPC(batch, painter, texture, new RandomBehavior(4));
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