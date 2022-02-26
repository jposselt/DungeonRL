import com.badlogic.gdx.graphics.g2d.SpriteBatch;

import graphic.Painter;
import interfaces.IEntity;
import tools.Point;

public class GenericNPC<T> implements IEntity {
    private SpriteBatch batch;
    private Painter painter;
    private String texture;
    private Point position;
    private IBehavior<T> behavior;

    public GenericNPC(SpriteBatch batch, Painter painter, String texture, IBehavior<T> behavior){
        this.batch=batch;
        this.painter=painter;
        this.texture = texture;
        this.behavior = behavior;
    }

    @Override
    public SpriteBatch getBatch() {
        return batch;
    }

    @Override
    public Painter getPainter() {
        return painter;
    }

    @Override
    public Point getPosition() {
        return position;
    }

    @Override
    public String getTexture() {
        return texture;
    }

    @Override
    public boolean removable() {
        return false;
    }

    @Override
    public void update() {
        this.draw();
    }

    public IBehavior<T> getBehavior() {
        return behavior;
    }

    public void setPostion(Point p) {
        position = p;
    }
}
