package net.codejava.hibernate;

import javax.persistence.*;

@Entity
@Table(name="sense2Synset")
public class Sense2Synset{
	@EmbeddedId
	private Sense2SynsetID id;
	private Boolean main;
	
	public Sense2Synset() {
		super();
		// TODO Auto-generated constructor stub
	}

	public Sense2Synset(Sense2SynsetID id, Boolean main) {
		super();
		this.id = id;
		this.main = main;
	}

	public Sense2SynsetID getId() {
		return id;
	}

	public void setId(Sense2SynsetID id) {
		this.id = id;
	}

	public Boolean isMain() {
		return main;
	}

	public void setMain(Boolean main) {
		this.main = main;
	}

	@Override
	public String toString() {
		return "Sense2Synset [id=" + id + ", main=" + main + "]";
	}
	
}
