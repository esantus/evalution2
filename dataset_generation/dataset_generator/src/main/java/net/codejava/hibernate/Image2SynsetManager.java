package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;

public class Image2SynsetManager {

    public void create(Long synset_id, Long image_id, Long source_id, Boolean main, SessionFactory sessionFactory) {
        Image2SynsetID i2sID = new Image2SynsetID();
    	Image2Synset i2s = new Image2Synset();
    	
    	i2sID.setSynset_id(synset_id);
    	i2sID.setImage_id(image_id);
    	i2s.setId(i2sID);
    	i2s.setMain(main);
    	i2s.setSource_id(source_id);
    	
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
        
        if(session.get(Image2Synset.class, i2sID) == null){
        	session.save(i2s);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long synset_id, Long image_id, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        Image2SynsetID i2sID = new Image2SynsetID();
        i2sID.setImage_id(image_id);
        i2sID.setSynset_id(synset_id);
        
        Image2Synset w2syn = session.get(Image2Synset.class, i2sID);
     
        System.out.println("Id: " + w2syn.getId());
        System.out.println("Source: " + w2syn.getSource_id());
        System.out.println("Main: "+w2syn.isMain());
        
        session.close();
    }
 
    public void update(Long synset_id, Long image_id, Long source_id, Boolean main, SessionFactory sessionFactory) {
        Image2SynsetID i2sID = new Image2SynsetID();
    	Image2Synset i2s = new Image2Synset();
    	
    	i2sID.setSynset_id(synset_id);
    	i2sID.setImage_id(image_id);
    	i2s.setId(i2sID);
    	i2s.setMain(main);
    	i2s.setSource_id(source_id);
    	   
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(i2s);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synset_id, Long image_id, SessionFactory sessionFactory) {
        // code to remove a book
        Image2SynsetID i2sID = new Image2SynsetID();
    	Image2Synset i2s = new Image2Synset();

    	i2sID.setImage_id(image_id);
    	i2sID.setSynset_id(synset_id);
    	i2s.setId(i2sID);
        
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(i2s);
     
        session.getTransaction().commit();
        session.close();
    }
}