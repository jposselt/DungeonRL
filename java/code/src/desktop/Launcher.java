package desktop;

import com.badlogic.gdx.backends.lwjgl.DungeonApplication;
import com.badlogic.gdx.backends.lwjgl.LwjglApplicationConfiguration;
import controller.LibgdxSetup;
import controller.MainController;
import tools.Constants;

public final class Launcher {
    public static void run(MainController mc) {
        LwjglApplicationConfiguration config = new LwjglApplicationConfiguration();
        config.width = Constants.WINDOW_WIDTH;
        config.height = Constants.WINDOW_HEIGHT;
        config.foregroundFPS = 3;
        new DungeonApplication(new LibgdxSetup(mc), config, 0);
    }
}