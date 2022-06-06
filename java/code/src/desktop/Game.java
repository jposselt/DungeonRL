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
    private final String levelFile;
    private final String modelPath;

    public Game(String level, String model) {
        this.levelFile = level;
        this.modelPath = model;
    }

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
        if (npc.getPosition().toCoordinate().equals(levelAPI.getCurrentLevel().getEndTile().getCoordinate())) {
            npc.setPostion(getRandomPosition());
        }
    }

    protected Point getRandomPosition() {
        Random rand = new Random();
        List<Room> rooms = levelAPI.getCurrentLevel().getRooms();
        Room room = rooms.get(rand.nextInt(rooms.size()));
        return room.getRandomFloorTile().getCoordinate().toPoint();
    }

    @Override
    protected void setup() {
        LevelLoader loader = new LevelLoader();
        loader.loadLevel(this.levelFile);
        levelAPI.setGenerator(loader);

        final String texture = "character/monster/chort_idle_anim_f0.png";
        IBehavior<Point, Integer> behavior = new TFModel(this.modelPath);
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
        final String model = "";
        final String level = "";
        Launcher.run(new Game(level, model));
    }
}