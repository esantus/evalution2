package net.codejava.hibernate;
import javax.persistence.*;

@Entity
@Table(name="synsetID")
public class SynsetID {
	private Long synset_id;
	private String synset_value;
	
	
	public SynsetID() {
		super();
		// TODO Auto-generated constructor stub
	}
	
	public SynsetID(Long synset_id, String synset_value) {
		super();
		this.synset_id = synset_id;
		this.synset_value = synset_value;
	}
	
	@Id
	@Column(name = "synset_id")
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	public Long getSynset_id() {
		return synset_id;
	}

	public void setSynset_id(Long synset_id) {
		this.synset_id = synset_id;
	}

	public String getSynset_value() {
		return synset_value;
	}

	public void setSynset_value(String synset_value) {
		this.synset_value = synset_value;
	}

	@Override
	public String toString() {
		return "SynsetID [synset_id=" + synset_id + ", synset_value=" + synset_value + "]";
	}
	
}
