package desktop;

import controller.MainController;
import level.LevelAPI;
import level.generator.dummy.DummyGenerator;
import level.generator.dungeong.graphg.NoSolutionException;

public class Game extends MainController {

    @Override
    public void onLevelLoad() {
        // TODO Auto-generated method stub
        
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

        //entityController.add(hero);
        //camera.follow();

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