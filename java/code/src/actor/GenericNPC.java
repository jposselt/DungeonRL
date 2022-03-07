package actor;

import behavior.IBehavior;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import graphic.Painter;
import interfaces.IEntity;
import level.elements.Level;
import tools.Point;

public class GenericNPC<I,O> implements IEntity {
    private SpriteBatch batch;
    private Painter painter;
    private String texture;
    private Point position;
    private Level level;
    private IBehavior<I,O> behavior;

    public GenericNPC(SpriteBatch batch, Painter painter, String texture, IBehavior<I,O> behavior){
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

    public IBehavior<I,O> getBehavior() {
        return behavior;
    }

    public void setPostion(Point p) {
        position = p;
    }

    public void setLevel(Level level) {
        this.level = level;
        this.position = level.getStartTile().getGlobalPosition().toPoint();
    }

    public Level getLevel() {
        return level;
    }
}