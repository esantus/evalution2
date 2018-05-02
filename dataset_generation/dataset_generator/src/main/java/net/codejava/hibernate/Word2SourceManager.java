package net.codejava.hibernate;

import org.hibernate.Session;
import org.hibernate.SessionFactory;


public class Word2SourceManager {
	
    public void create(Long synset_id, Long word_id, Long language_id, Long source_id, SessionFactory sessionFactory) {
        Word2SourceID w2sID = new Word2SourceID();
        Word2Source w2s = new Word2Source();
        
        w2sID.setSynset_id(synset_id);
        w2sID.setWord_id(word_id);
        w2sID.setLanguage_id(language_id);
        w2sID.setSource_id(source_id);
        
        w2s.setId(w2sID);
        
        
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        if(session.get(Word2Source.class, w2sID) == null){
        	session.save(w2s);
        }
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void read(Long synset_id, Long word_id, Long language_id, Long source_id, SessionFactory sessionFactory) {
        // code to get a book
        Session session = sessionFactory.openSession();
        Word2SourceID w2sID = new Word2SourceID();
        w2sID.setSynset_id(synset_id);
        w2sID.setWord_id(word_id);
        w2sID.setLanguage_id(language_id);
        w2sID.setSource_id(source_id);
        
        Word2Source w2s = session.get(Word2Source.class, w2sID);
        
        System.out.println("Id: " + w2s.getId());
        
        session.close();
    }
 
    public void update(Long synset_id, Long word_id, Long language_id, Long source_id, SessionFactory sessionFactory) {
        Word2SourceID w2sID = new Word2SourceID();
        Word2Source w2s = new Word2Source();
        
        w2sID.setSynset_id(synset_id);
        w2sID.setWord_id(word_id);
        w2sID.setLanguage_id(language_id);
        w2sID.setSource_id(source_id);
        
        w2s.setId(w2sID);
        
        Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.update(w2s);
     
        session.getTransaction().commit();
        session.close();
    }
 
    public void delete(Long synset_id, Long word_id, Long language_id, Long source_id, SessionFactory sessionFactory) {
        Word2SourceID w2sID = new Word2SourceID();
        Word2Source w2s = new Word2Source();
        
        w2sID.setSynset_id(synset_id);
        w2sID.setWord_id(word_id);
        w2sID.setLanguage_id(language_id);
        w2sID.setSource_id(source_id);
        
        w2s.setId(w2sID);
        
    	Session session = sessionFactory.openSession();
        session.beginTransaction();
     
        session.delete(w2s);
     
        session.getTransaction().commit();
        session.close();
    }
}