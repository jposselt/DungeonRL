package actor;

import behavior.IBehavior;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import graphic.Painter;
import tools.Point;

import java.util.List;

public class RandomNPC extends GenericNPC<List<Integer>, Integer> {
    public RandomNPC(SpriteBatch batch, Painter painter, String texture, IBehavior<List<Integer>, Integer> behavior) {
        super(batch, painter, texture, behavior);
    }

    @Override
    public void update() {
        Point newPosition = new Point(this.getPosition());
        float movementSpeed = 1f;

        int action = getBehavior().nextAction(null);
        if (action == 0) newPosition.y += movementSpeed;
        if (action == 1) newPosition.y -= movementSpeed;
        if (action == 2) newPosition.x += movementSpeed;
        if (action == 3) newPosition.x -= movementSpeed;
        if(this.getLevel().getTileAt(newPosition.toCoordinate()).isAccessible()) {
            this.setPostion(newPosition);
        }

        this.draw();
    }
}
