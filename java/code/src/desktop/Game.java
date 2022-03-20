package desktop;

import actor.TestNPC;
import behavior.IBehavior;
import behavior.TFModel;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input;
import controller.MainController;
import level.elements.room.Room;
import level.generator.LevelLoader.LevelLoader;
import level.generator.dungeong.graphg.NoSolutionException;
import tools.Point;

import java.util.List;
import java.util.Random;


public class Game extends MainController {
    private TestNPC npc;

    @Override
    public void onLevelLoad() {
        npc.setLevel(levelAPI.getCurrentLevel());
    }

    @Override
    protected void beginFrame() {
        if (Gdx.input.isKeyPressed(Input.Keys.R)) {
            npc.setPostion(getRandomPosition());
        }
    }

    @Override
    protected void endFrame() {
        if (npc.getPosition().toCoordinate().equals(levelAPI.getCurrentLevel().getEndTile().getGlobalPosition())) {
            npc.setPostion(getRandomPosition());
        }
    }

    protected Point getRandomPosition() {
        Random rand = new Random();
        List<Room> rooms = levelAPI.getCurrentLevel().getRooms();
        Room room = rooms.get(rand.nextInt(rooms.size()));
        return room.getRandomFloorTile().getGlobalPosition().toPoint();
    }

    @Override
    protected void setup() {
        levelAPI.setGenerator(new LevelLoader());

        final String texture = "character/monster/chort_idle_anim_f0.png";
        final String modelDir = "D:\\Repos\\DungeonRL\\models\\tf\\ppo-1000";
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