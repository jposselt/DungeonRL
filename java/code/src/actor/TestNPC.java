package actor;

import behavior.IBehavior;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import graphic.Painter;
import tools.Point;

public class TestNPC extends GenericNPC<Point, Integer>{

    public TestNPC(SpriteBatch batch, Painter painter, String texture, IBehavior<Point, Integer> behavior) {
        super(batch, painter, texture, behavior);
    }

    private Point nextPosition(int action) {
        Point p = new Point(this.getPosition());
        float movementSpeed = 1f;
        if (action == 0) p.y += movementSpeed;
        if (action == 1) p.y -= movementSpeed;
        if (action == 2) p.x -= movementSpeed;
        if (action == 3) p.x += movementSpeed;

        return p;
    }

    private boolean isValidAction(int action) {
        return this.getLevel().getTileAt(nextPosition(action).toCoordinate()).isAccessible();
    }

    private boolean[] getActionMask() {
        return new boolean[] {
                isValidAction(0),
                isValidAction(1),
                isValidAction(2),
                isValidAction(3)
        };
    }

    @Override
    public void update() {
        int action = getBehavior().nextAction(this.getPosition(), getActionMask());
        this.setPostion(nextPosition(action));
        this.draw();
    }
}
